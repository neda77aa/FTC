# -*- coding: utf-8 -*-
from network import Network
from get_config import get_config
from dataloader.as_dataloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive

import wandb

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="contrastive learning", entity="asproject",config = config, name = 'tad_1e-4_tclloss_batch16')
    
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr = get_as_dataloader(config, split='train', mode='train')
    dataloader_ssl = get_as_dataloader(config, split='train_all', mode='ssl')
    dataloader_va = get_as_dataloader(config, split='val', mode='val')
    dataloader_test = get_as_dataloader(config, split='test', mode='val')
    dataloader_te = get_as_dataloader(config, split='test', mode='test')
    dataloader_validation = get_as_dataloader(config, split='val', mode='test')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va,dataloader_test)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="ssl":
        net.train(dataloader_ssl, dataloader_va)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test_comprehensive(dataloader_validation, mode="test", record_embeddings=False)
    if config['use_wandb']:
        wandb.finish()