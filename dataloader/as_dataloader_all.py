#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']


label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1}
}
class_labels: Dict[str, List[str]] = {
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['mild', 'moderate', 'severe'],
    'mild_moderate': ['mild', 'moderate'],
    'moderate_severe': ['moderate', 'severe']
}

    
def get_as_dataloader(config, split, mode):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    droot = r"/mnt/nas-server/datasets/cardiac/processed/aortic-stenonsis"
    
    if mode=='train':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
        show_info = False
    if mode=='ssl':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
        show_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        bsize = config['batch_size']
        show_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        bsize = 1
        show_info = True
        
    if show_info:
        assert bsize==1, "To show per-data info batch size must be 1"
    if config['model'] == 'slowfast':
        fr = 32
    else:
        fr = 16
        
    dset = AorticStenosisDataset(dataset_root=droot, 
                                split=split,
                                view=config['view'],
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                contrastive_method = config['cotrastive_method'],
                                flip_rate=flip,
                                label_scheme_name=config['label_scheme_name'])
    
    if mode=='train':
        if config['sampler'] == 'AS':
            sampler_AS, _ = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS)
        elif config['sampler'] == 'bicuspid':
            _ , sampler_B = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_B)
        else: # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=True)
    return loader
    

class AorticStenosisDataset(Dataset):
    def __init__(self, dataset_root: str = '~/as', view: str = 'plax',
                 split: str = 'train',
                 transform: bool = True, normalize: bool = True, 
                 frames: int = 16, resolution: int = 224,
                 cine_loader: str = 'mat_loader', return_info: bool = False,
                 contrastive_method: str = 'CE',
                 flip_rate: float = 0.3, min_crop_ratio: float = 0.8, 
                 hr_mean: float = 4.237, hr_std: float = 0.1885,
                 label_scheme_name: str = 'all',
                 **kwargs):
        # if normalize: # TODO normalization might be key to improving accuracy
        #     raise NotImplementedError('Normalization is not yet supported, data will be in the range [0-1]')

        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])
        
        # read in the data directory CSV as a pandas dataframe
        
        #dataset = pd.read_csv(join(dataset_root, 'annotations-all.csv'))
        dataset = pd.read_csv('/AS_Neda/AS_thesis/aortic_stenosis.csv')
        
        # append dataset root to each path in the dataframe
        # tip: map(lambda x: x+1) means add 1 to each element in the column
        dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
        
        if view in ('plax', 'psax'):
            dataset = dataset[dataset['view'] == view]
        elif view != 'all':
            raise ValueError(f'View should be plax, psax or all, got {view}')
       
        # remove unnecessary columns in 'as_label' based on label scheme
        self.scheme = label_schemes[label_scheme_name]
        dataset = dataset[dataset['as_label'].isin( self.scheme.keys() )]
        # # modify those values to their numerical counterparts
        # dataset['as_label'] = dataset['as_label'].map(lambda x: self.scheme[x])

        self.cine_loader = get_loader(cine_loader)
        self.return_info = return_info
        self.hr_mean = hr_mean
        self.hr_srd = hr_std

        # Take train/test/val
        if split in ('train', 'val', 'test', 'ulb'):
            dataset = dataset[dataset['split'] == split]
        elif split == 'train_all':
            dataset = dataset[dataset['split'].isin(['train','ulb'])]
        elif split != 'all':
            raise ValueError(f'View should be train/val/test/all, got {split}')

        self.dataset = dataset
        self.frames = frames
        self.resolution = (resolution, resolution)
        self.split = split
        self.transform = None
        self.transform_contrastive = None
        self.pack_transform = PackPathway(alpha=4)
        if transform:
            self.transform = Compose(
                [RandomResizedCropVideo(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlipVideo(p=flip_rate)]
            )
        if contrastive_method!= 'CE':
            self.transform_contrastive = Compose(
                [RandomResizedCropVideo(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlipVideo(p=flip_rate)]
            )
            
        self.normalize = normalize
        self.contrstive = contrastive_method

    def class_samplers(self):
        # returns WeightedRandomSamplers
        # based on the frequency of the class occurring
        
        # storing labels as a dictionary will be in a future update
        labels_B = np.array(self.dataset['Bicuspid'])*1
        # storing labels as a dictionary will be in a future update
        labels_AS = np.array(self.dataset['as_label'])  
        labels_AS = np.array([self.scheme[t] for t in labels_AS])
        class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
                                          for t in np.unique(labels_AS)])
        weight_AS = 1. / class_sample_count_AS
        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS,0,0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        #samples_weight_AS = samples_weight_AS.double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        if labels_B[0] == labels_B[0]:
            class_sample_count_B = np.array([len(np.where(labels_B == t)[0]) 
                                             for t in np.unique(labels_B)])
            weight_B = 1. / class_sample_count_B
            samples_weight_B = np.array([weight_B[t] for t in labels_B])

            samples_weight_B = torch.from_numpy(samples_weight_B).double()
            #samples_weight_B = samples_weight_B.double()
            sampler_B = WeightedRandomSampler(samples_weight_B, len(samples_weight_B))
        else:
            sampler_B = 0
        return sampler_AS, sampler_B
        

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor-m)/std

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]

        cine_original = self.cine_loader(data_info['path'])
        folder = data_info['Study_Folder']
        folder = 'round2'
        if folder == 'all_cines':
            cine_original = cine_original.transpose((2,0,1))
        elif folder == 'round2':
            pass
            
        window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
        cine = self.get_random_interval(cine_original, window_length)
        #print(cine.shape)
        #cine = resize(cine, (self.frames, *self.resolution))
        cine = resize(cine, (32, *self.resolution))
        cine = torch.tensor(cine).unsqueeze(0)
        
        # storing labels as a dictionary will be in a future update
        if folder == 'round2':
            #labels_B = torch.tensor(int(data_info['Bicuspid']))
            labels_B = torch.tensor(int(2))
        if folder == 'all_cines':
            labels_B = torch.tensor(int(2))
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])

        if self.transform:
            if self.contrstive == 'CE' or self.contrstive == 'Linear':
                cine = self.transform(cine)
            else:
                cine_org = self.transform(cine)
                cine_aug = self.transform_contrastive(cine)
                cine = cine_org
                if random.random() < 0.4:
                    upsample = nn.Upsample(size=(16,224, 224), mode='nearest')
                    cine_aug = cine_aug[:, :,  0:180, 40:180].unsqueeze(1)
                    cine_aug = upsample(cine_aug).squeeze(1)   
                
        if self.normalize:
            cine = self.bin_to_norm(cine)
            if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='all'):
                cine_aug = self.bin_to_norm(cine_aug)  

        cine = self.gray_to_gray3(cine)
        cine = cine.float()
        
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            cine_aug = self.gray_to_gray3(cine_aug)
            cine_aug = cine_aug.float()
            
        
        # slowFast input transformation
        #cine = self.pack_transform(cine)
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            ret = ([cine,cine_aug], labels_AS, labels_B)
       
        else:
            ret = (cine, labels_AS, labels_B)
        if self.return_info:
            di = data_info.to_dict()
            di['window_length'] = window_length
            di['original_length'] = cine_original.shape[1]
            ret = (cine, labels_AS, labels_B, di, cine_original)

        return ret

# extra transformation for slowfast architecture    
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha=8):
        self.alpha = alpha
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list