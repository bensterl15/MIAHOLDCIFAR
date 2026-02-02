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

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
sqrt6 = np.sqrt(6)

hold_T = 5.0

class HOLD(nn.Module):
    def __init__(self, config, L_inv=0.5, alpha = 0.08, is_image=True, hold_objective = 'hsm', numerical_eps=1e-9, device='cuda:0'):
        
        n = config.model_order
        # nOLD specific parameters:
        self.lambdastar = -math.sqrt(2*n - 3)
        
        # Initialize diagonal of Sigma_0
        Sigma_0_diag = torch.full((n,), alpha*L_inv, device=device).double()
        Sigma_0_diag[0] = 0
        self.Sigma_0_diag = Sigma_0_diag-L_inv

        # Compute gammas and xi
        gammas = torch.tensor([(n*n - i*i) / (4*i*i - 1) for i in range(1, n)], device=device)
        self.gammas = -self.lambdastar * torch.flip(gammas, dims=[0]).sqrt()
        xi = -n * self.lambdastar
        
        # Construct F
        F_matrix = torch.diag(self.gammas, 1) - torch.diag(self.gammas, -1)
        F_matrix[-1, -1] = -xi
        self.F_matrix = torch.kron(F_matrix, torch.eye(3, device=device)).double()

        # precompute taylor expansion coefficient matrices of expFt
        F_shifted = F_matrix - self.lambdastar*torch.eye(n, device=device)
        F_shifted = F_shifted.double()
        F_k = torch.eye(n, device=device).double()
        Bs = [F_k]
        for k in range(1, n):
            F_k = F_k @ F_shifted / k
            Bs.append(F_k)

        self.Bs = torch.stack(Bs, dim=-1)
    
        self.n = n
        self.L_inv = L_inv
        self.xi = xi
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
    
    def sde(self, x, t):
        """
        Evaluating drift and diffusion of the SDE.
        """
        drift = torch.einsum("ij,bj...->bi...", self.F_matrix, x.double())
        d = drift.shape[1] // self.n
        diffusion = torch.zeros_like(drift)
        diffusion[:, ((self.n-1)*d):((self.n)*d)] = np.sqrt(2. * self.xi * self.L_inv) * torch.ones_like(diffusion[:, ((self.n-1)*d):((self.n)*d)])

        return drift, diffusion
    
    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(x, t, score=None):
            """
            Evaluating drift and diffusion of the ReverseSDE.
            """
            
            drift, diffusion = sde_fn(x, hold_T - t)
            score = score if score is not None else score_fn(x, hold_T - t)
            
            d = drift.shape[1] // self.n
            
            diffusion_s = diffusion[:, ((self.n-1)*d):((self.n)*d)]

            reverse_drift = -drift
            reverse_drift[:, ((self.n-1)*d):((self.n)*d)] = reverse_drift[:, ((self.n-1)*d):((self.n)*d)] + diffusion_s ** 2. * score * (0.5 if probability_flow else 1.)

            reverse_diffusion = torch.zeros_like(diffusion)
            if not probability_flow:
                reverse_diffusion[:, ((self.n-1)*d):((self.n)*d)] = diffusion_s
            
            return reverse_drift, reverse_diffusion

        return reverse_sde
    
    def prior_sampling(self, shape):
        priors = [torch.randn(*shape, device=self.config.device) * np.sqrt(self.L_inv) for _ in range(self.n)]
        return torch.cat(priors, dim=1)

    def prior_logp(self, x):
        N = np.prod(x.shape[1:]) # the dimension of feature maps
        logx = -N / 2. * np.log(2. * np.pi * self.L_inv) - torch.sum(x.view(x.shape[0], -1) ** 2., dim=1) / (2. * self.L_inv)
        return logx
        
    def mean(self, x, expFt):
        '''
        Evaluating the mean of the conditional perturbation kernel.
        '''
        d = x.shape[1] // self.n
        expFt_ = torch.kron(expFt, torch.eye(d, device=self.device)).double()
        mu_t = torch.einsum("bik,bk...->bi...", expFt_, x.double())
        return mu_t
    
    #### Last function:
    def var(self, expFt):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        Sigma_t = torch.einsum("bik,k,bjk->bij", expFt, self.Sigma_0_diag, expFt)
        Sigma_t.diagonal(dim1=-2, dim2=-1).add_(self.L_inv + self.numerical_eps)
        return Sigma_t

    def mean_and_var(self, x, expFt):
        return self.mean(x, expFt), self.var(expFt)

    @property
    def is_augmented(self):
        return True

    def noise_multiplier(self, t_):
        '''
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        '''
        t = add_dimensions(t_, False).double()
        
        powers = t ** torch.arange(self.n, device=self.device).double()
        expFt = torch.einsum(
            "bk,ijk->bij",
            (torch.exp(self.lambdastar * t) * powers),
            self.Bs.double()  # make sure Bs is also double
        )
        
        Sigma_t = self.var(expFt)
        # Take Kronecker product so dimensions match:
        Sigma_t = torch.kron(Sigma_t, torch.eye(self.d, device=self.device)).double()
        L_t = self.cholesky_unroll(Sigma_t)
        L_t = torch.nan_to_num(L_t, nan=self.numerical_eps).double()
        coeff = -1 / L_t[:, -1, -1]
        
        # Add on the extra dimensions:
        coeff = add_dimensions(coeff, self.is_image)
        
        return coeff

    def perturb_data(self, batch, t_, B_ATTACKER=False, model=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and
        \gamma * M for HSM.
        '''        
        if B_ATTACKER:
            t = add_dimensions(t_, False).double()

            powers = t ** torch.arange(self.n, device=self.device).double()
            expFt = torch.einsum(
                "bk,ijk->bij",
                (torch.exp(self.lambdastar * t) * powers),
                self.Bs.double()
            )

            # Case of zeros:
            padding = torch.zeros((batch.shape[0], self.d * (self.n - 1), batch.shape[2], batch.shape[3]),dtype=batch.dtype, device=batch.device)
            # Initial noise:
            #padding = self.alpha * self.L_inv * torch.randn((batch.shape[0], 3 * (self.n - 1), batch.shape[2], batch.shape[3]),dtype=batch.dtype, device=batch.device)            
            batch = torch.cat([batch, padding], dim=1).type(torch.float32)

            d = batch.shape[1] // self.n
            self.d = d
            self.batch_size = batch.shape[0]

            # Preallocate outputs
            mean_all = torch.zeros_like(batch, dtype=torch.float64)
            batch_randn = torch.zeros_like(batch, dtype=torch.float64)
            noise_all = torch.zeros_like(batch, dtype=torch.float64)

            processing_batch_size = 512
            endpt = batch.shape[0] # 2 * processing_batch_size#
            for i in tqdm(range(0, endpt, processing_batch_size)):
                batch_chunk = batch[i:i + processing_batch_size].float()
                expFt_chunk = expFt[i:i + processing_batch_size]
                #t_chunk = t[i:i + processing_batch_size].float().squeeze()

                mean_chunk, Sigma_t_chunk = self.mean_and_var(batch_chunk, expFt_chunk)
                Sigma_t_chunk = torch.kron(Sigma_t_chunk, torch.eye(d, device=self.device)).double()
                L_t_chunk = self.cholesky_unroll(Sigma_t_chunk)
                L_t_chunk = torch.nan_to_num(L_t_chunk, nan=0.0).double()

                # Estimate score
                with torch.no_grad():
                    # t = 0 is really not stable.. Plug in numerical epsilon:
                    zeros_t = self.numerical_eps * torch.ones(batch_chunk.shape[0], device=batch.device, dtype=torch.float32)
                    score_chunk = model(batch_chunk, zeros_t)

                eps_est_chunk = -score_chunk * L_t_chunk[:, -1, -1].view(-1, 1, 1, 1)
                batch_randn_chunk = torch.zeros_like(batch_chunk, dtype=torch.float64)
                #batch_randn_chunk = torch.randn_like(batch_chunk, dtype=torch.float64)
                batch_randn_chunk[:, (self.d * (self.n - 1)):(self.d * self.n)] = eps_est_chunk

                noise_chunk = torch.einsum("bij,bj...->bi...", L_t_chunk, batch_randn_chunk)

                # Store results
                mean_all[i:i + processing_batch_size] = mean_chunk
                batch_randn[i:i + processing_batch_size] = batch_randn_chunk
                noise_all[i:i + processing_batch_size] = noise_chunk

                #print(f'L_t stats: min={L_t_chunk.min()}, max={L_t_chunk.max()}, mean={L_t_chunk.mean()}, std={L_t_chunk.std()}', flush=True)
                #print(f'score_est stats: min={score_chunk.min()}, max={score_chunk.max()}, mean={score_chunk.mean()}, std={score_chunk.std()}', flush=True)
                #print(f'noise stats: min={noise_chunk.min()}, max={noise_chunk.max()}, mean={noise_chunk.mean()}, std={noise_chunk.std()}', flush=True)

                # Cleanup
                del batch_chunk, expFt_chunk, mean_chunk, Sigma_t_chunk
                del L_t_chunk, score_chunk, eps_est_chunk, batch_randn_chunk, noise_chunk
                torch.cuda.empty_cache()

            perturbed_data = mean_all + noise_all
            return perturbed_data
        else:
            t = add_dimensions(t_, False).double()
            #print(t_.shape, flush=True)
            #print(t.shape,)

            powers = t ** torch.arange(self.n, device=self.device).double()
            expFt = torch.einsum(
                "bk,ijk->bij",
                (torch.exp(self.lambdastar * t) * powers),
                self.Bs.double()  # make sure Bs is also double
            )
            mean, Sigma_t = self.mean_and_var(batch, expFt)
            d = batch.shape[1] // self.n
            self.d = d
            self.batch_size = batch.shape[0]
            
            # Take Kronecker product so dimensions match:
            Sigma_t = torch.kron(Sigma_t, torch.eye(d, device=self.device)).double()
            L_t = self.cholesky_unroll(Sigma_t)
            L_t = torch.nan_to_num(L_t, nan=0.0).double()

            batch_randn = torch.randn_like(batch, device=batch.device).double()
            noise = torch.einsum("bij,bj...->bi...", L_t, batch_randn)        
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
    
    def cholesky_unroll(self, A):
        """
        Unrolled Cholesky decomposition for arbitrary-size SPD matrices.
        A: (B, n, n) batch of SPD matrices
        Returns: (B, n, n) lower-triangular matrices L such that A ≈ LLᵀ
        """
        B, n, _ = A.shape
        L = torch.zeros_like(A, device=A.device)

        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    sum_k = torch.sum(L[:, i, :j] ** 2, dim=1)
                    L[:, i, j] = torch.sqrt(A[:, i, i] - sum_k)
                else:
                    sum_k = torch.sum(L[:, i, :j] * L[:, j, :j], dim=1)
                    L[:, i, j] = (A[:, i, j] - sum_k) / L[:, j, j]
        return L
