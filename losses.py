# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from util.utils import add_dimensions
from models import utils as mutils
from sde_lib import hold_T

def get_loss_fn(sde, train, config):
    def loss_fn(model, x):
        n = config.model_order
        # Setting up initial means
        if sde.is_augmented:
            if config.cld_objective == 'dsm':
                if sde.is_hold:
                    noise_terms = [torch.randn_like(x, device=x.device) * np.sqrt(sde.alpha / sde.L) for _ in range(n - 1)]
                    batch = torch.cat([x] + noise_terms, dim=1)
                else:
                    v = torch.randn_like(x, device=x.device) * \
                        np.sqrt(sde.gamma / sde.m_inv)
                    batch = torch.cat((x, v), dim=1)
            elif config.cld_objective == 'hsm':
                # For HSM we are marginalizing over the full initial velocity
                if sde.is_hold:
                    zero_terms = [torch.zeros_like(x, device=x.device) for _ in range(n - 1)]
                    batch = torch.cat([x] + zero_terms, dim=1)

                else:
                    v = torch.zeros_like(x, device=x.device)
                    batch = torch.cat((x, v), dim=1)
            else:
                raise NotImplementedError(
                    'The objective %s for CLD-SGMs is not implemented.' % config.cld_objective)
        else:
            batch = x

        if sde.is_hold:
            t = torch.rand(batch.shape[0], device=batch.device,
                           dtype=torch.float64) * (hold_T - config.loss_eps) + config.loss_eps
        else:
            t = torch.rand(batch.shape[0], device=batch.device,
                           dtype=torch.float64) * (1.0 - config.loss_eps) + config.loss_eps

        perturbed_data, mean, _, batch_randn = sde.perturb_data(batch, t)
        perturbed_data = perturbed_data.type(torch.float32)
        mean = mean.type(torch.float32)

        # In the augmented case, we only need "velocity noise" for the loss
        if sde.is_augmented:
            if sde.is_hold:
                batch_randn_s = torch.chunk(batch_randn, n, dim=1)[-1]
                batch_randn = batch_randn_s
            else:
                _, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)
                batch_randn = batch_randn_v


        score_fn = mutils.get_score_fn(config, sde, model, train)
        score = score_fn(perturbed_data, t)
        #print(perturbed_data.shape, flush=True)
        #print(t.shape, flush=True)

        if sde.is_hold:
            multiplier = 1.0
        else:
            multiplier = sde.loss_multiplier(t).type(torch.float32)
            multiplier = add_dimensions(multiplier, config.is_image)

        noise_multiplier = sde.noise_multiplier(t).type(torch.float32)

        if config.weighting == 'reweightedv1':
            loss = (score / noise_multiplier - batch_randn)**2 * multiplier
        elif config.weighting == 'likelihood':
            # Following loss corresponds to Maximum Likelihood learning
            loss = (score - batch_randn * noise_multiplier)**2 * multiplier
        elif config.weighting == 'reweightedv2':
            loss = (score / noise_multiplier - batch_randn)**2
        else:
            raise NotImplementedError(
                'The loss weighting %s is not implemented.' % config.weighting)

        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        if torch.sum(torch.isnan(loss)) > 0:
            raise ValueError(
                'NaN loss during training; if using CLD, consider increasing config.numerical_eps')

        return loss
    return loss_fn


def get_step_fn(train, optimize_fn, sde, config):
    loss_fn = get_loss_fn(sde, train, config)

    scaler = GradScaler() if config.autocast_train else None

    def step_fn(state, batch, optimization=True):
        model = state['model']

        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            with autocast(enabled=config.autocast_train):
                loss = torch.mean(loss_fn(model, batch))

            if optimization:
                if config.autocast_train:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                optimize_fn(optimizer, model.parameters(),
                            state['step'], scaler)
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                with autocast(enabled=config.autocast_eval):
                    loss = torch.mean(loss_fn(model, batch))
        return loss
    return step_fn
