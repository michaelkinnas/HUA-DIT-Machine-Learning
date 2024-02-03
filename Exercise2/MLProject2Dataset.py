import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import glob
import os
from pathlib import Path
from helper_functions import vallidation


class MLProject2Dataset(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform=None, subset=None, meta=False):
        self.data_dir = data_dir
        self.transform = transform
        self.meta = meta
        
        # Create dataframe containing image id and file paths
        self.files = pd.DataFrame({
            'image_id': [Path(path).stem for path in glob.glob(f'{data_dir}/*/*')],
            'path': [os.path.abspath(path) for path in glob.glob(f'{data_dir}/*/*')]
        })

        # Create dataframe for metadata
        self.metadata = pd.read_csv(data_dir + '/' + metadata_fname)
        self.metadata['dx'] = self.metadata['dx'].astype('category').cat.codes
        
        sex = pd.get_dummies(self.metadata['sex'], prefix='sex', dtype=float)
        loc = pd.get_dummies(self.metadata['localization'], prefix='loc', dtype=float)

        # Process metadata columns
        self.metadata = pd.concat([self.metadata, sex, loc], axis=1)
        self.metadata = self.metadata.drop(['sex', 'localization'], axis=1)
        self.metadata['age'] = self.metadata['age'] / 100

        # Drop some age NaN values and re-index
        self.metadata = self.metadata.dropna()

        # Create main dataframe
        # self.dataset = pd.merge(self.files, self.metadata[['dx', 'image_id']], on='image_id')    
        self.dataset = pd.merge(self.files, self.metadata.drop(['lesion_id', 'dx_type'], axis=1), on='image_id')
       

        # Smaller dataset size
        if subset == 'tiny':
            self.dataset = self.dataset.sample(n=200, axis = 0)
            self.dataset.reset_index(drop=True, inplace=True)
        elif subset == 'small':
            self.dataset = self.dataset.sample(n=1000, axis = 0)
            self.dataset.reset_index(drop=True, inplace=True)
        elif subset == 'half':
            self.dataset = self.dataset.sample(frac=0.5, axis = 0)
            self.dataset.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        image = torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255
        if self.transform:
            image = self.transform(image)

        if self.meta == True:
            return (image, torch.tensor(self.dataset.loc[idx, 'age':], dtype=torch.float), self.dataset.at[idx, 'dx'].astype('long'))
        else:
            return (image, self.dataset.at[idx, 'dx'].astype('long'))