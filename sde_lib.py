# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
## Modified:

import math
import torch
import torch.nn as nn
import numpy as np
from util.utils import add_dimensions

from tqdm import tqdm

hold_T = 1.0

class HOLD(nn.Module):
    def __init__(self, config, L_inv=0.5, alpha = 0.08, is_image=True, hold_objective = 'hsm', numerical_eps=1e-9, device='cuda:0'):
        
        n = config.model_order
        # nOLD specific parameters:
        self.lambdastar = -math.sqrt(2*n - 3) if n > 1 else -1.0
        
        self.beta0 = config.beta0
        self.beta1 = config.beta1
            
        self.n = n
        self.alpha = alpha # the scaling of the variance in the initial p and s distribution
        self.is_image = is_image
        self.hold_objective = hold_objective
        self.numerical_eps = numerical_eps
        self.config = config
        self.device=device
        
        # Temporary placeholder:
        self.batch_size = 0
        # Set to 3 because RGB are 3 colors:
        self.d = 3

    def beta(self, t):
        return self.beta0 + (self.beta1 - self.beta0) * t

    def B(self, t):
        return self.beta0*t + 0.5*(self.beta1 - self.beta0)*t**2

    def sde(self, x, t):
        """
        Evaluating drift and diffusion of the SDE.
        """
        drift = -self.beta(t)*x/2 #torch.einsum("ij,bj...->bi...", self.F_matrix, x.double())
        diffusion = torch.sqrt(self.beta(t))
        return drift, diffusion
    
    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(x, t_, score=None):
            """
            Evaluating drift and diffusion of the ReverseSDE.
            """
            t = add_dimensions(t_, self.is_image).double()
            drift, diffusion = sde_fn(x, hold_T - t)
            score = score if score is not None else score_fn(x, hold_T - t_)
            
            reverse_drift = - drift + diffusion ** 2. * score * (0.5 if probability_flow else 1.)

            reverse_diffusion = torch.zeros_like(diffusion)
            if not probability_flow:
                reverse_diffusion = diffusion
            
            return reverse_drift, reverse_diffusion

        return reverse_sde
    
    def prior_sampling(self, shape):
        priors = [torch.randn(*shape, device=self.config.device) for _ in range(self.n)]
        return torch.cat(priors, dim=1)

    def prior_logp(self, x):
        N = np.prod(x.shape[1:]) # the dimension of feature maps
        logx = -N / 2. * np.log(2. * np.pi) - torch.sum(x.view(x.shape[0], -1) ** 2., dim=1) / (2.)
        return logx
        
    def mean(self, x, t):
        '''
        Evaluating the mean of the conditional perturbation kernel.
        '''
        #print(f'x.shape = {x.shape}, t.shape = {t.shape}', flush=True)
        return x * torch.exp(-self.B(t)/2)
    
    #### Last function:
    def var(self, t):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        return 1 - torch.exp(-self.B(t))

    def mean_and_var(self, x, t):
        return self.mean(x, t), self.var(t)

    @property
    def is_augmented(self):
        return True

    def noise_multiplier(self, t_):
        '''
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        '''
        t = add_dimensions(t_, self.is_image).double()
        l_t = torch.sqrt(self.var(t)) + self.numerical_eps
        if torch.any(torch.isnan(l_t)):
            raise ValueError('NaN detected in booboo 1')
        coeff = -1 / l_t
        #coeff = add_dimensions(coeff, self.is_image)
        if torch.any(torch.isnan(coeff)):
            raise ValueError('NaN detected in booboo 2')
        return coeff

    def perturb_data(self, batch, t_, B_ATTACKER=False, model=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and
        \gamma * M for HSM.
        '''        
        if B_ATTACKER:
            t = add_dimensions(t_, self.is_image).double()

            self.d = batch.shape[1]
            self.batch_size = batch.shape[0]

            # Preallocate outputs
            mean_all = torch.zeros_like(batch, dtype=torch.float64)
            batch_randn = torch.zeros_like(batch, dtype=torch.float64)
            noise_all = torch.zeros_like(batch, dtype=torch.float64)

            processing_batch_size = 512
            endpt = batch.shape[0]
            for i in tqdm(range(0, endpt, processing_batch_size)):
                batch_chunk = batch[i:i + processing_batch_size].float()
                t_chunk = t[i:i + processing_batch_size]

                mean_chunk, Sigma_t_chunk = self.mean_and_var(batch_chunk, t_chunk)
                l_t_chunk = torch.sqrt(Sigma_t_chunk)

                # Estimate score
                with torch.no_grad():
                    # t = 0 is really not stable.. Plug in numerical epsilon:
                    zeros_t = self.numerical_eps * torch.ones(batch_chunk.shape[0], device=batch.device, dtype=torch.float32)
                    score_chunk = model(batch_chunk, zeros_t)

                eps_est_chunk = -score_chunk * l_t_chunk.view(-1, 1, 1, 1)
                noise_chunk = l_t_chunk * eps_est_chunk

                # Store results
                mean_all[i:i + processing_batch_size] = mean_chunk
                batch_randn[i:i + processing_batch_size] = eps_est_chunk
                noise_all[i:i + processing_batch_size] = noise_chunk

                #print(f'L_t stats: min={L_t_chunk.min()}, max={L_t_chunk.max()}, mean={L_t_chunk.mean()}, std={L_t_chunk.std()}', flush=True)
                #print(f'score_est stats: min={score_chunk.min()}, max={score_chunk.max()}, mean={score_chunk.mean()}, std={score_chunk.std()}', flush=True)
                #print(f'noise stats: min={noise_chunk.min()}, max={noise_chunk.max()}, mean={noise_chunk.mean()}, std={noise_chunk.std()}', flush=True)

                # Cleanup
                del batch_chunk, mean_chunk, Sigma_t_chunk
                del l_t_chunk, score_chunk, eps_est_chunk, noise_chunk
                torch.cuda.empty_cache()

            perturbed_data = mean_all + noise_all
            return perturbed_data
        else:
            t = add_dimensions(t_, self.is_image).double()

            mean, Sigma_t = self.mean_and_var(batch, t)
            self.d = batch.shape[1]
            self.batch_size = batch.shape[0]

            batch_randn = torch.randn_like(batch, device=batch.device).double()
            noise = torch.sqrt(Sigma_t) * batch_randn      
            perturbed_data = mean + noise
            return perturbed_data, mean, noise, batch_randn

    @property
    def is_hold(self):
        return True

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(x, t, dt):
            vec_t = torch.ones(
                x.shape[0], device=x.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(x, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn(*x.shape, device=x.device)

            x_mean = x + drift
            x = x_mean + diffusion * noise
            return x, x_mean

        return discrete_step_fn
