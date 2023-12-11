import torch
import torchvision
import pandas as pd
import glob
import os
from pathlib import Path
import PIL

class MLProject2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Create dataframe containing image id and file paths
        self.files = pd.DataFrame({
            'image_id': [Path(path).stem for path in glob.glob(f'{data_dir}/*/*')],
            'path': [os.path.abspath(path) for path in glob.glob(f'{data_dir}/*/*')]
        })

        # Create dataframe for metadata
        self.metadata = pd.read_csv(data_dir + '/'+metadata_fname)
        self.metadata['dx'] = self.metadata['dx'].astype('category').cat.codes
        

        # Create main dataframe
        self.dataset = pd.concat([self.files['path'],  self.metadata['dx']], axis=1)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return (torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255, self.dataset.at[idx, 'dx'])
      
