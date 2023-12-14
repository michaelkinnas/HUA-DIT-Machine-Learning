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
        
        image = torchvision.io.read_image(self.dataset.at[idx, 'path']).to(torch.float32) / 255
        if self.transform:
            image = self.transform(image)

        return (image, self.dataset.at[idx, 'dx'].astype('long'))
      

def image_transforms(m, n):
    return transforms.Compose([
                transforms.Resize((m, n), antialias=True),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])


# TODO accuracy
def train_fn(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, 
          epochs: int = 10, optimizer: optim = None, loss: nn.modules.loss = None,
          device: str = 'cpu', print_period: int = 10) -> None:
    
    print(f'Training on {device}')
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        current_period_total = 0
        for batch, (X, y) in enumerate(trainloader, 0):
            # Move to device
            X, y = X.to(device), y.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(X)
            current_loss = loss(pred, y)

            # Convert probs to prediction
            yhat = torch.argmax(pred, 1)
        
            # Count correct predictions
            train_correct += (yhat == y).type(torch.float).sum().item()
            current_period_total += y.size(0)

            # Backpropagation
            current_loss.backward()

            # Parameter update
            optimizer.step()

            # Print            
            running_loss += current_loss.item()

            if batch % print_period == print_period-1:
                avg_loss = running_loss / print_period

                model.eval()
                with torch.inference_mode():
                    valid_running_loss = 0.0
                    valid_correct = 0
                    for (X, y) in valloader:
                        X, y = X.to(device), y.to(device)
                        # Model predictions
                        pred_valid = model(X)

                        # Calculate loss for current batch
                        valid_running_loss += loss(pred_valid, y).item()

                        # Convert probs to prediction
                        yhat = torch.argmax(pred, 1)

                        # Count correct predictions
                        # valid_correct += (yhat == y).type(torch.float).sum().item()

                print(f"Epoch: {epoch}, batch: {batch:5d}] train loss: {avg_loss:.3f} train acc: {train_correct/current_period_total:.3f} | val loss: {valid_running_loss / len(valloader.dataset):.3f} val acc: {valid_correct/len(valloader.dataset):.3f}")
                
                running_loss = 0.0
                # Put model back into train mode
                model.train()
                


def test_net(model: nn.Module, testloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> None:
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        test_running_loss = 0.0
        for (X, y) in testloader:        
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # Calculate loss
            test_running_loss += loss(pred, y).item() 

            # Convert to predictions and calculate accuracy
            yhat = torch.argmax(pred, 1)
            total += y.size(0)
            correct += (yhat == y).type(torch.float).sum().item()

    print(f"Average loss: {test_running_loss/len(testloader.dataset):.4f}. Test accuracy in {total} images: {correct/total:4f}")  #Chech the 4f parameter without dot