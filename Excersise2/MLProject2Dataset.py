import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
import torchvision
import pandas as pd
import glob
import os
from pathlib import Path
import PIL

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
        return (torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255, self.dataset.at[idx, 'dx'])
      

def image_transforms(m, n):
    return transforms.Compose([
                transforms.Resize((m, n)),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

# def train(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, 
#           epochs: int = 10, optimizer: optim = None, loss: nn.modules.loss = None,
#           device: str = 'cpu', print_period: int = 10) -> None:
    
    
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Compute gradients with backpropagation
#         loss.backward()
#         # Execute an optimization step
#         optimizer.step()
#         # Need to zero out our gradients, since loss.backward() accumulates losses
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch+1)*len(X)
#             print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")