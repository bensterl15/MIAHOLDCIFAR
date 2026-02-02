import os
import glob
import logging
import time
import torch
from torch.utils import tensorboard
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import gc

from util.utils import calculate_frechet_distance
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from models import ncsnpp
from util.utils import make_dir, get_optimizer, optimization_manager, get_data_scaler, get_data_inverse_scaler, set_seeds, save_img
from util.utils import compute_eval_loss, compute_image_likelihood, broadcast_params, reduce_tensor, build_beta_fn, build_beta_int_fn
from util import datasets
from util.checkpoint import save_checkpoint, restore_checkpoint
import losses
import sde_lib
import sampling
import likelihood
import evaluation

import matplotlib.pyplot as plt
import tempfile
from io import BytesIO
from PIL import Image
import wandb
import gc
from tqdm import tqdm

## Convenience function to collect all the images from each data loader:
def collect_all_images(data_loader, device=None):
    all_images = []
    for x, _ in data_loader:
        if device is not None:
            x = x.to(device)
        all_images.append(x)
    return torch.cat(all_images, dim=0)

def run_proximal_inference_attack(config, model, sde):
    # 1 channel because MNIST is grayscale:
    n_channels = 3
    n_discrete_steps = 10
    hold_T = 5.0
    delta_t = hold_T / n_discrete_steps
    N_ROC_points = 100

    # Load the training and validation loaders:
    train_loader, val_loader, _ = datasets.get_loaders(config)

    x_train = collect_all_images(train_loader, device=config.device)
    x_val   = collect_all_images(val_loader, device=config.device)

    # Only take first 1000 of each category:
    x_train = x_train[:1000]
    x_val = x_val[:1000]

    # Optional: build labels for ROC
    labels = torch.cat([
        torch.ones(x_train.shape[0], dtype=torch.uint8),
        torch.zeros(x_val.shape[0], dtype=torch.uint8)
    ], dim=0)

    processing_batch_size = 512     # Do this to prevent OOM errors:

    x_0 = torch.cat([x_train, x_val], dim=0)
    end_pt = x_0.shape[0] # 2 * processing_batch_size#
    ### TEMPORARY:
    x_0 = x_0[:end_pt]
    labels = labels[:end_pt]
    ###
    data_size = x_0.shape[0]
    R_tp = torch.zeros(data_size, n_discrete_steps, device=config.device)

    t = torch.zeros(data_size, device=config.device).double() + 0.01
    for t_ind in tqdm(range(n_discrete_steps)):
        x_t = sde.perturb_data(x_0, t, B_ATTACKER=True, model=model)
        x_t = x_t.detach()
        #print(f'x_t stats: min={x_t.min()}, max={x_t.max()}, mean={x_t.mean()}, std={x_t.std()}', flush=True)

        score_est_list = []
        for j in tqdm(range(0, end_pt, processing_batch_size)):
            print('', flush=True)
            batch_chunk = x_t[j:j+processing_batch_size]
            t_batch = t[j:j+processing_batch_size].squeeze(-1)
            with torch.no_grad():
                score_chunk = model(batch_chunk.float(), t_batch.float())
                #print(f'score_chunk.shape = {score_chunk.shape}', flush=True)
            score_est_list.append( score_chunk )
            del score_chunk
        score_est = torch.cat(score_est_list, dim=0)
        score_est = score_est.detach()
        print(f'score_est: {score_est.norm(dim=1).mean()}')
        #print(f'score_est stats: min={score_est.min()}, max={score_est.max()}, mean={score_est.mean()}, std={score_est.std()}', flush=True)
        #print(f'x_t stats: min={x_t.min()}, max={x_t.max()}, mean={x_t.mean()}, std={x_t.std()}', flush=True)

        F = sde.F_matrix
        print(f'F.shape={F.shape}', flush=True)
        print(f'x_t.shape={x_t.shape}', flush=True)
        R_tp_ = torch.einsum("ij,bjkl->bikl", F, x_t).cpu()
        score_est = score_est.cpu()
        R_tp_[:,((config.model_order-1)*n_channels):(config.model_order*n_channels)] = R_tp_[:,((config.model_order-1)*n_channels):(config.model_order*n_channels)] - sde.xi * sde.L_inv * score_est
        R_tp_ = R_tp_[:,((config.model_order-1)*n_channels):(config.model_order*n_channels)]
        R_tp[:, t_ind] = torch.norm(R_tp_.reshape(R_tp_.shape[0],-1), p=2, dim=1)

        # Call garbage collector to prevent OOM errors:
        del x_t, score_est, R_tp_
        gc.collect()
        torch.cuda.empty_cache()
        # Incrementing t at the end ensures hold_T - t is never zero!!!
        t += delta_t

    ## Sanity check:
    #R_tp[50000:60000] = R_tp[50000:60000] + 1e3 
    #R_tp = R_tp.mean(dim=1)

    #plt.plot(R_tp[:,0])
    #plt.savefig('R_tp.png')

    tau_min = R_tp.min()
    tau_max = R_tp.max()
    taus = torch.linspace(tau_min, tau_max, N_ROC_points)

    print(R_tp, flush=True)

    ROC_sizes = torch.zeros(N_ROC_points)
    ROC_powers = torch.zeros(N_ROC_points)
    for n_ROC in range(N_ROC_points):
        tau = taus[n_ROC]
        b_vect = (R_tp < tau).float().mean(dim=1) > 0.5
        b_vect_powers = b_vect[labels==1]
        b_vect_sizes = b_vect[labels==0]
        ROC_sizes[n_ROC] = b_vect_sizes.sum()/b_vect_sizes.shape[0]
        ROC_powers[n_ROC] = b_vect_powers.sum()/b_vect_powers.shape[0]

    # Approximate the Area under the curve with trapezoidal integration:
    AUC = np.trapz(ROC_powers, ROC_sizes)

    print(f'AUC = {AUC}', flush=True)

    #plt.rcParams['figure.figsize'] = [20, 5]
    #plt.plot(ROC_sizes, ROC_powers)
    #plt.plot(ROC_sizes, ROC_sizes, 'r--')
    #plt.title(f'ROC AUC={AUC}')

    # Save to a buffer and convert to PIL image for WandbLogger
    #buf = BytesIO()
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #image = Image.open(buf)
    #wandb.log({"ROC": wandb.Image(image)})

    # Log NumPy array directly from memory
    #ROC_data = np.column_stack((ROC_sizes, ROC_powers))
    #with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
    #    np.save(tmp, ROC_data)
    #    tmp.flush()
    #    
    #    artifact = wandb.Artifact(name="ROC_data", type='dataset')
    #    artifact.add_file(tmp.name)
    #    wandb.log_artifact(artifact)

    wandb.log({"val/AUROC": AUC})

    #plt.close()
