# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy import integrate

from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from model.utils import fix_len_compatibility

import os
import logging
import re
log = logging.getLogger()

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path='./config')
def main(cfg: DictConfig):
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")
     
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device(f'cuda:{cfg.training.gpu}')
    
    log.info('Initializing logger...')
    log_dir = cfg.training.tensorboard_dir
    logger = SummaryWriter(log_dir=log_dir)

    log.info('Initializing data loaders...')
    train_dataset = TextMelDataset('train', cfg)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=cfg.training.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=cfg.training.num_workers, shuffle=False)
    test_dataset = TextMelDataset('dev', cfg)

    log.info('Initializing model...')
    model = GradTTS(cfg)
    model.to(device)
    out_size = fix_len_compatibility(2*cfg.data.sample_rate//256)

    log.info('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))
    log.info('Number of decoder parameters = %.2fm' % (model.decoder.nparams/1e6))


############################################################################################
    checkpoint = torch.load(cfg.training.pre_checkpoint)
    print(f'cfg.training.pre_checkpoint is {cfg.training.pre_checkpoint}')
    match = re.search(r'_([\d]+)\.pt', cfg.training.pre_checkpoint)
    if match:
        value = int(match.group(1))  # Extract the first capture group and convert to integer
    value = value + 1 if value != 0 else value
    
    if cfg.training.load_encoder==True :
        if cfg.training.load_decoder==True :
            keys = {key: value for key, value in checkpoint.items()}
            model.load_state_dict(keys, strict=False)
        else:
            encoder_keys = {key: value for key, value in checkpoint.items() if key.startswith('encoder')}
            model.load_state_dict(encoder_keys, strict=False)

############################################################################################
    

    log.info('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)

    log.info('Start training...')
    iteration = 0
    
    os.makedirs(f"{cfg.training.checkpoint_save}", exist_ok=True)

    for epoch in range(value, cfg.training.n_epochs):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset)//cfg.training.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar): #cause alot of warning for stft with return_complex=False
                model.zero_grad()
                x, x_lengths = batch['x'].to(device), batch['x_lengths'].to(device)
                y, y_lengths = batch['y'].to(device), batch['y_lengths'].to(device)
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                
                #####################################################################                
                #####################################################################
                
                #with torch.autograd.detect_anomaly(check_nan=True):
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}\n'
                    progress_bar.set_description(msg)
                
                iteration += 1
        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)
        
        log.info(msg)
        
        if epoch % cfg.training.save_every > 0:
            continue
        print('I am in this place')
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{cfg.training.checkpoint_save}/grad_{epoch}.pt")

if __name__ =='__main__':
    main()
