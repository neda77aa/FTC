import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import tmed_dataloader

droot = "/AS_clean/TMED/approved_users_only/"

dset = tmed_dataloader.TMEDDataset(dataset_root=droot, 
                                    split='all',
                                    view='all',
                                    transform=True,
                                    normalize=True,
                                    flip_rate=0.5,
                                    label_scheme_name='five_class')

as_loader = DataLoader(dset, batch_size=16, shuffle=True)

d = next(iter(as_loader))
x = d['x']
y_view = d['y_view']
y_AS = d['y_AS']

x0 = x[0]
x0 = x0.numpy()
plt.imshow(x0.transpose(1,2,0))

# calculate (a very approximate) mean and variance
d_a = [] 
for i, d in enumerate(as_loader):
    if i < 100:
        d_a.append(np.mean(d['x'].numpy()))
    if i > 100:
        break
    
d_avg = np.mean(d_a)

d_v =  []
for i, d in enumerate(as_loader):
    if i < 100:
        d_v.append(np.mean((d['x'].numpy()-d_avg)**2))
    if i > 100:
        break
    
d_variance = np.mean(d_v)
d_std = np.sqrt(d_variance)
    