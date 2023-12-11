import torch
import pandas as pd
import glob
import os
from pathlib import Path

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
        self.metadata = pd.read_csv(metadata_fname)
        self.metadata['dx'] = self.metadata['dx'].astype('category').cat.codes
        

        # Create main dataframe
        self.dataset = pd.concat([self.files,  self.metadata['dx']], axis=1)
        
        
