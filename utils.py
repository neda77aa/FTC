from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from dataloader.as_dataloader import get_as_dataloader
from tqdm import tqdm
import faiss
import torch.nn as nn


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def compute_embeddings(loader, model, embed_dim):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, embed_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)
    for idx, (cine, labels,_) in enumerate(tqdm(loader)):
        cine = cine.cuda()
        bsz = labels.shape[0]

        embed = model(cine)
        total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
        total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del cine, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def validation_constructive(model,config):
    valid_loader = get_as_dataloader(config, split='val', mode='val')
    config_c = config.copy()
    config_c['cotrastive_method'] = 'CE'
    config_c['batch_size'] = 2
    train_loader = get_as_dataloader(config_c, split='train', mode='train')
    calculator = AccuracyCalculator(k=1)
    model.eval()
    
    query_embeddings, query_labels = compute_embeddings(valid_loader, model,config['feature_dim'])
    print('Done')
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model,config['feature_dim'])
    print('Done_2')
    acc_dict = calculator.get_accuracy(
        query_embeddings,
        reference_embeddings,
        query_labels,
        reference_labels,
        embeddings_come_from_same_source=False
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()
    model.train()

    return acc_dict

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x