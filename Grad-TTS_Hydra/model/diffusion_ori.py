# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
import numpy as np
from scipy.fftpack import dct, idct
from torch.distributions.multivariate_normal import MultivariateNormal
from einops import rearrange
import matplotlib.pyplot as plot
import os
from model.base import BaseModule
from model.utils import Cutout
from model.utils import cutout_along_dimension

def pt_to_pdf(pt, pdf, vmin=-12.5, vmax=0.0):
    spec = pt
    fig = plot.figure(figsize=(20, 4), tight_layout=True)
    subfig = fig.add_subplot()
    image = subfig.imshow(spec, cmap="jet", origin="lower", aspect="equal", interpolation="none", vmax=vmax,
                          vmin=vmin)
    fig.colorbar(mappable=image, orientation='vertical', ax=subfig, shrink=0.5)
    plot.savefig(pdf, format="pdf")
    plot.close()

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        x = torch.clamp(x, min=-1e3, max=1e3)
        output = self.block(x * mask)
        return output * mask

class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(ResnetBlock, self).__init__()
        self.block1 = Block(dim,     dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()
        )

    def forward(self, x, mask):
        h = self.block1(x, mask)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        x = torch.clamp(x, min=-1e3, max=1e3)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            


        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = torch.clamp(x, min=-1e3, max=1e3)
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(BaseModule):
    def __init__(
        self, 
        dim, 
        dim_mults=(1, 2, 4), 
        groups=8, 
        n_spks=None, 
        spk_emb_dim=64, 
        n_feats=80, 
        dropout_rate=0.1
    ):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        # If n_spks is None, default to 1 (single speaker)
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim

        # Speaker embedding MLP (only if using multiple speakers)
        if self.n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats)
            )

        # Determine channel dimensions at each resolution
        # Starts with 2 channels (mu and x) or 3 channels (mu, x, and s) if multi-speaker
        dims = [2 + (1 if self.n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Downsampling layers
        self.downs = torch.nn.ModuleList([])
        # Upsampling layers
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, groups=self.groups),
                torch.nn.Dropout(dropout_rate),
                ResnetBlock(dim_out, dim_out, groups=self.groups),
                torch.nn.Dropout(dropout_rate),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()
            ]))
        
        # Middle (bottleneck) layers
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, groups=self.groups)
        self.mid_block1_dropout = torch.nn.Dropout(dropout_rate)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, groups=self.groups)
        self.mid_block2_dropout = torch.nn.Dropout(dropout_rate)

        # Upsampling layers (mirror of downsampling)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, groups=self.groups),
                torch.nn.Dropout(dropout_rate),
                ResnetBlock(dim_in, dim_in, groups=self.groups),
                torch.nn.Dropout(dropout_rate),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)
            ]))

        # Final output layers
        self.final_block = Block(dim, dim)
        self.final_block_dropout = torch.nn.Dropout(dropout_rate)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, spk=None):
        # Speaker conditioning (if applicable)
        if spk is not None:
            s = self.spk_mlp(spk)
        # Stack input channels: [mu, x] (and speaker embedding s if multi-speaker)
        if self.n_spks < 2:
            x = torch.stack([mu, x], dim=1)
        else:
            # Repeat speaker embedding across time dimension and stack as third channel
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], dim=1)

        # Expand mask to match the number of input channels
        mask = mask.unsqueeze(1)
        hiddens = []   # list to save activations for skip connections
        masks = [mask]  # track masks at each resolution for skip connections

        # Downsampling path
        for resnet1, dropout1, resnet2, dropout2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down)       # ResNet block 1 (no time embedding)
            x = dropout1(x)                 # apply dropout
            x = resnet2(x, mask_down)       # ResNet block 2 (no time embedding)
            x = dropout2(x)                 # apply dropout
            x = attn(x)                     # attention layer (Residual connection inside)
            hiddens.append(x)               # save hidden activation for skip connection
            # Downsample (if not last layer) and update mask for next layer
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])  # downsample mask (halve spatial dimensions)

        # Prepare for mid (bottleneck) layers
        masks = masks[:-1]              # drop the mask corresponding to final downsampling (not used further)
        mask_mid = masks[-1]            # mask at bottleneck

        # Bottleneck (middle) layers
        x = self.mid_block1(x, mask_mid)    # ResNet block in bottleneck (no time embedding)
        x = self.mid_block1_dropout(x)
        x = self.mid_attn(x)               # attention in bottleneck
        x = self.mid_block2(x, mask_mid)    # second ResNet block in bottleneck (no time embedding)
        x = self.mid_block2_dropout(x)

        # Upsampling path
        for resnet1, dropout1, resnet2, dropout2, attn, upsample in self.ups:
            mask_up = masks.pop()          # retrieve corresponding mask from downsampling path
            # Concatenate skip connection from corresponding downsampling layer
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up)        # ResNet block 1 during upsampling (no time embedding)
            x = dropout1(x)
            x = resnet2(x, mask_up)        # ResNet block 2 during upsampling (no time embedding)
            x = dropout2(x)
            x = attn(x)                    # attention layer
            x = upsample(x * mask_up)      # upsample and apply mask

        # Final output stage
        x = self.final_block(x, mask)
        x = self.final_block_dropout(x)
        output = self.final_conv(x * mask)
        return (output * mask).squeeze(1)



def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise

class Diffusion(BaseModule):
    def __init__(self, cfg):
        super(Diffusion, self).__init__()
        self.n_spks = cfg.data.n_spks
        self.spk_emb_dim = cfg.model.spk_emb_dim
        self.n_feats = cfg.data.n_feats
        
        self.dim = cfg.model.decoder.dim
        self.pe_scale = cfg.model.decoder.pe_scale
        
        self.n_timesteps = cfg.training.n_timesteps
        cfg = cfg.model.Masking
        self.beta_min = cfg.a
        self.beta_max = cfg.b
        self.c = cfg.c
        self.d = cfg.d
        self.estimator = GradLogPEstimator2d(self.dim, n_spks=self.n_spks,
                                             spk_emb_dim=self.spk_emb_dim,
                                             dropout_rate = self.d)



    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        os.makedirs(f'reverse_pass', exist_ok=True)
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask

            ################################
            #i = t[0].item()
            #pt_to_pdf(xt[0].cpu(), f'reverse_pass/Xn_{i}.pdf' , vmin=-12.5, vmax=0.0)


        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        
        dropout = torch.nn.Dropout(p=self.d)
        mu_dropout = dropout(mu)

        noise_estimation = self.estimator(xt, mask, mu_dropout, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)
