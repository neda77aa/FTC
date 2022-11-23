# -*- coding: utf-8 -*-
from network_p import Network
from get_config import get_config
from dataloader.tmed_patientloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive

import wandb

if __name__ == "__main__":
    
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="TMED", entity="asproject",config = config, name = 'wrn-50-3-class')
    
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr_4 = get_as_dataloader(config, split='train', mode='train',start=4,finish=8)
    dataloader_tr_8 = get_as_dataloader(config, split='train', mode='train',start=8,finish=16)
    dataloader_tr_16 = get_as_dataloader(config, split='train', mode='train',start=16,finish=20)
    dataloader_tr_20 = get_as_dataloader(config, split='train', mode='train',start=20,finish=24)
    dataloader_tr_24 = get_as_dataloader(config, split='train', mode='train',start=24,finish=32)
    dataloader_tr_32 = get_as_dataloader(config, split='train', mode='train',start=32,finish=48)
    config['batch_size'] = 2
    dataloader_tr_48 = get_as_dataloader(config, split='train', mode='train',start=48,finish=64)
    dataloader_tr_64 = get_as_dataloader(config, split='train', mode='train',start=64,finish=128)
    dataloader_tr = [dataloader_tr_4,dataloader_tr_8,dataloader_tr_16,dataloader_tr_20,dataloader_tr_24,dataloader_tr_32,dataloader_tr_48,dataloader_tr_64]
    dataloader_va = get_as_dataloader(config, split='val', mode='val')
    dataloader_test = get_as_dataloader(config, split='test', mode='val')
    dataloader_te = get_as_dataloader(config, split='val', mode='test')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va,dataloader_test)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="ssl":
        net.train(dataloader_ssl, dataloader_va)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test_comprehensive(dataloader_te, mode="test", record_embeddings=True)
    if config['use_wandb']:
        wandb.finish()