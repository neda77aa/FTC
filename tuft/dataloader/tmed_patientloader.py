#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader#, WeightedRandomSampler
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip

from random import seed

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

'''
Dataloader for fully supervised learning of TMED2
2 modes, loading on a patient-level and loading on an image-level
Performs data augmentation transforms
'''


DATA_ROOT = "/AS_clean/TMED/approved_users_only/"
CSV_NAME = 'DEV479/TMED2_fold0_labeledpart.csv'
# SOCKEYE: DATA_ROOT = TBD
# LOCAL: DATA_ROOT = "D:\\Datasets\\TMED\\approved_users_only"

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader, 'png_loader': png_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    return loadmat(path)['cine']

def png_loader(path):
    return plt.imread(path)


label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 1},
    'mild_mod': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 2},
    'mod_severe': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 1, 'severe_AS': 2},
    'four_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 1, 'moderate_AS': 2, 'severe_AS': 3},
    'five_class': {'no_AS': 0, 'mild_AS': 1, 'mildtomod_AS': 2, 'moderate_AS': 3, 'severe_AS': 4},
}

view_scheme = {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4}
view_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'three_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':2},
    'four_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':2, 'A4CorA2CorOther':3},
    'five_class': {'PLAX':0, 'PSAX':1, 'A2C':2, 'A4C':3, 'A4CorA2CorOther':4},
}


#For human reference
class_labels: Dict[str, List[str]] = {
    'binary': ['No AS', 'AS'],
    'mild_mod': ['No AS', 'Early', 'Significant'],
    'mod_severe': ['No AS', 'Mild-mod', 'Severe'],
    'four_class': ['No AS', 'Mild', 'Moderate', 'Severe'],
    'five_class': ['No AS', 'Mild', 'Mild-mod', 'Moderate', 'Severe']
}


def get_as_dataloader(config, split, mode,start=8,finish=16):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test'/'all' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    droot = DATA_ROOT
    
    if mode=='train':
        flip = 0.5
        tra = True
        bsize = config['batch_size']
        patient_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        start = 0
        finish = 200
        bsize = 1
        patient_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        start = 0
        finish = 200
        bsize = 1
        patient_info = True
        
    dset = TMEDDataset(dataset_root=droot, 
                        split=split,
                        view=config['view'],
                        transform=tra,
                        normalize=True,
                        flip_rate=flip,
                        patient_info = patient_info, 
                        start = start,
                        finish = finish,
                        label_scheme_name=config['label_scheme_name'],
                        view_scheme_name=config['view_scheme_name']
                         )
    # TODO change class samplers to be based on patient frequency
    # sampler_AS, sampler_B = dset.class_samplers()
    
    if mode=='train':
        # if config['sampler'] == 'AS':
        #     loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS)
        # elif config['sampler'] == 'bicuspid':
        #     loader = DataLoader(dset, batch_size=bsize, sampler=sampler_B)
        # else: # random sampling
        #     loader = DataLoader(dset, batch_size=bsize, shuffle=True)
        loader = DataLoader(dset, batch_size=bsize, shuffle=True)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=False)
    return loader

# TODO get a searchable version of the dataset with all splits for running quick visualizations


class TMEDDataset(Dataset):
    def __init__(self, 
                 dataset_root: str = '~/as',
                 view: str = 'PLAX', # PLAX/PSAX/PLAXPSAX/no_other/all
                 split: str = 'train', # train/val/test/'all'
                 transform: bool = True, 
                 normalize: bool = False, 
                 resolution: int = 224,
                 image_loader: str = 'png_loader', 
                 flip_rate: float = 0.5,  
                 patient_info: bool = False,
                 start: int = 16,
                 finish: int = 32, 
                 label_scheme_name: str = 'all', # see above
                 view_scheme_name: str = 'three_class',
                 **kwargs):
        # navigation for linux environment
        # dataset_root = dataset_root.replace('~', os.environ['HOME'])
        self.dataset_root = dataset_root
        
        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(join(dataset_root, CSV_NAME))
        # append dataset root to each path in the dataframe
        dataset['path'] = dataset.apply(self.get_data_path_rowwise, axis=1)
        
        if view in ('PLAX', 'PSAX'):
            dataset = dataset[dataset['view_label'] == view]
        elif view == 'plaxpsax':
            dataset = dataset[dataset['view_label'].isin(['PLAX', 'PSAX'])]
        elif view == 'no_other':
            dataset = dataset[dataset['view_label'] != 'A4CorA2CorOther']
        elif view != 'all':
            raise ValueError(f'View should be PLAX/PSAX/PLAXPSAX/no_other/all, got {view}')
       
        # remove unnecessary columns in 'as_label' based on label scheme
        self.scheme = label_schemes[label_scheme_name]
        self.scheme_view = view_schemes[view_scheme_name]
        dataset = dataset[dataset['diagnosis_label'].isin( self.scheme.keys() )]
        # # modify those values to their numerical counterparts
        # dataset['as_label'] = dataset['as_label'].map(lambda x: self.scheme[x])

        self.image_loader = get_loader(image_loader)
        self.patient_info = patient_info

        # Take train/test/val
        if split in ('train', 'val', 'test'):
            dataset = dataset[dataset['diagnosis_classifier_split'] == split]
        elif split != 'all':
            raise ValueError(f'View should be train/val/test/all, got {split}')
            
           
        self.start = start
        self.finish = finish
        p_id = dataset['query_key'].str.split("_", n = 1, expand = True)
        dataset["p_id"] = p_id[0]
        patient_list_in = np.unique(dataset["p_id"])
        patient_list =[]
        for patient_id in patient_list_in:
            data_patient = dataset[dataset["p_id"]==patient_id]
            if (len(data_patient)<self.finish) and (len(data_patient)>=self.start):
                patient_list.append(patient_id)
        
        self.dataset = dataset
        self.patient_list = patient_list
        self.resolution = (resolution, resolution)

        self.transform = None
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(0.8, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
        self.normalize = normalize

    # def class_samplers(self):
    #     # returns WeightedRandomSamplers
    #     # based on the frequency of the class occurring
        
    #     # storing labels as a dictionary will be in a future update
    #     labels_B = np.array(self.dataset['Bicuspid'])*1
    #     # storing labels as a dictionary will be in a future update
    #     labels_AS = np.array(self.dataset['as_label'])  
    #     labels_AS = np.array([self.scheme[t] for t in labels_AS])

    #     class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
    #                                       for t in np.unique(labels_AS)])
    #     weight_AS = 1. / class_sample_count_AS
    #     samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
    #     samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
    #     #samples_weight_AS = samples_weight_AS.double()
    #     sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))

    #     class_sample_count_B = np.array([len(np.where(labels_B == t)[0]) 
    #                                      for t in np.unique(labels_B)])
    #     weight_B = 1. / class_sample_count_B
    #     samples_weight_B = np.array([weight_B[t] for t in labels_B])

    #     samples_weight_B = torch.from_numpy(samples_weight_B).double()
    #     #samples_weight_B = samples_weight_B.double()
    #     sampler_B = WeightedRandomSampler(samples_weight_B, len(samples_weight_B))
    #     return sampler_AS, sampler_B
        

    def __len__(self) -> int:
        return len(self.patient_list)

    # get a dataset path from the TMED2 CSV row
    def get_data_path_rowwise(self, pdrow):
        path = join(self.dataset_root, pdrow['SourceFolder'], pdrow['query_key'])
        return path

    def get_random_interval(self, vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    def gray_to_gray3(self, in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    def bin_to_norm(self, in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.061
        std = 0.140
        return (in_tensor-m)/std
    
    def _get_image(self, data_info):
        '''
        General method to get an image and apply tensor transformation to it

        Parameters
        ----------
        data_info : ID of the item to retrieve (based on the position in the dataset)
            DESCRIPTION.

        Returns
        -------
        ret : size 3xTxHxW tensor representing image
            if return_info is true, also returns metadata

        '''

        img_original = self.image_loader(data_info['path'])
        
        img = resize(img_original, self.resolution) # HxW
        x = torch.tensor(img).unsqueeze(0) # 1xHxW
    

        if self.transform:
            x = self.transform(x)
        if self.normalize:
            x = self.bin_to_norm(x)

        x = self.gray_to_gray3(x)
        x = x.float() # 3xHxW
        
        return x
    
    def _get_patient(self, p_id):
        images = []
        y_view = []
        y_AS = []
        data_patient = self.dataset[self.dataset["p_id"]==p_id]
        len_min = self.start if self.start>0 else len(data_patient)
        for i in range(len_min):
            data_info = data_patient.iloc[i]
            images.append(self._get_image(data_info))
            y_view.append(torch.tensor(self.scheme_view[data_info['view_label']]))
            y_AS.append(torch.tensor(self.scheme[data_info['diagnosis_label']]))
        images = torch.stack(images)
        y_view = torch.stack(y_view)
        ret = {'x':images, 'y_AS':y_AS[0], 'y_view':y_view}
        
        return ret
        

    def __getitem__(self, item):
        p_id = self.patient_list[item]
        return self._get_patient(p_id)
    
    def tensorize_single_image(self, img_path):
        """
        Creates a video tensor that is consistent with config specifications
    
        Parameters
        ----------
        img_path : String
            the path to the image, the function will find matches of the path substring
    
        Returns
        -------
        see get_item_from_info
    
        """
        # look for a path in the dataset resembling the video path
        matches_boolean = self.dataset['path'].str.contains(img_path)
        found_entries=self.dataset[matches_boolean]
        if len(found_entries) == 0:
            raise ValueError('Found 0 matches for requested substring ' + img_path)
        elif len(found_entries) > 1:
            warnings.warn('Found multiple matches, returning first result')
        data_info = found_entries.iloc[0]
        return self._get_item_from_info(data_info)