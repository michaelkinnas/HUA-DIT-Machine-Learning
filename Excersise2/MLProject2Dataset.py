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
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Create dataframe containing image id and file paths
        self.files = pd.DataFrame({
            'image_id': [Path(path).stem for path in glob.glob(f'{data_dir}/*/*')],
            'path': [os.path.abspath(path) for path in glob.glob(f'{data_dir}/*/*')]
        })

        # Create dataframe for metadata
        self.metadata = pd.read_csv(data_dir + '/' + metadata_fname)
        self.metadata['dx'] = self.metadata['dx'].astype('category').cat.codes

        # Create main dataframe
        self.dataset = pd.merge(self.files, self.metadata[['dx', 'image_id']], on='image_id')
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        image = torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255
        if self.transform:
            image = self.transform(image)

        return (image, self.dataset.at[idx, 'dx'].astype('long'))
      


class MLProject2DatasetSmall(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Create dataframe containing image id and file paths
        self.files = pd.DataFrame({
            'image_id': [Path(path).stem for path in glob.glob(f'{data_dir}/*/*')],
            'path': [os.path.abspath(path) for path in glob.glob(f'{data_dir}/*/*')]
        })

        # Create dataframe for metadata
        self.metadata = pd.read_csv(data_dir + '/' + metadata_fname)
        self.metadata['dx'] = self.metadata['dx'].astype('category').cat.codes

        # Create main dataframe
        self.dataset = pd.merge(self.files, self.metadata[['dx', 'image_id']], on='image_id')
        self.dataset = self.dataset[:-9000]
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        image = torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255
        if self.transform:
            image = self.transform(image)

        return (image, self.dataset.at[idx, 'dx'].astype('long'))