import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import pandas as pd
import glob
import os
from pathlib import Path


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

#TODO return epoch metrics
def train_fn(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, 
          epochs: int = 10, optimizer: optim = None, loss: nn.modules.loss = None,
          device: str = 'cpu', print_period: int = 10) -> None:
    
    print(f'Training on {device}')
    
    epochs_report = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    model.train()
    for epoch in range(epochs):
        # Per epoch for returning and plotting
        epoch_accumulated_loss = 0.0
        epoch_train_correct = 0
        epoch_total = 0

        # Per period
        period_accumulated_loss = 0.0
        period_train_correct = 0
        period_total = 0

        for batch, (X, y) in enumerate(trainloader, 0):
            # Move to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            pred = model(X)
            current_loss = loss(pred, y)

            # Convert probs to prediction
            yhat = torch.argmax(pred, 1)
        
            # Count correct predictions
            period_train_correct += (yhat == y).type(torch.float).sum().item()
            epoch_train_correct += (yhat == y).type(torch.float).sum().item()
            period_total += y.size(0) # Add each batch size separatly
            epoch_total += y.size(0)

            # Zero gradients
            optimizer.zero_grad()
            # Backpropagation
            current_loss.backward()
            # Parameter update
            optimizer.step()
           
            period_accumulated_loss += current_loss.item()
            epoch_accumulated_loss += current_loss.item()

            # Print   
            if batch % print_period == 0:
                model.eval()
                with torch.inference_mode():
                    val_running_loss = 0.0
                    val_correct = 0
                    for v_batch, (X, y) in enumerate(valloader, 0):
                        X, y = X.to(device), y.to(device)

                        # Model predictions
                        pred_val = model(X)
                        current_loss = loss(pred_val, y)

                        # Calculate loss for current batch
                        val_running_loss += current_loss.item()
           
                        # Convert probs to prediction
                        val_yhat = torch.argmax(pred_val, 1)

                        # Count correct predictions
                        val_correct += (val_yhat == y).type(torch.float).sum().item()
                      
                #Printing period metrics
                avg_period_train_loss = period_accumulated_loss / print_period      # Accumulated loss / number of batches of reporting period
                train_period_accuracy = period_train_correct / period_total         # Accumulated acc / number of samples in the training batches of reporting period
                avg_val_loss = val_running_loss / (v_batch + 1)                     # Accumulated loss / number of validation batches
                val_accuracy = val_correct / len(valloader.dataset)                 # Accumulated acc / number of samples in validation dataset (all batches)

                print(f"[Epoch: {epoch}, batch: {batch:5d}] Train loss: {avg_period_train_loss:.3f}, Train acc: {train_period_accuracy:.3f} | Validation loss: {avg_val_loss:.3f}, Validation acc: {val_accuracy:.3f}")

                # Zero period counters
                period_accumulated_loss = 0.0
                period_train_correct = 0
                period_total = 0

                # Put model back into train mode
                model.train()
               
        # Calculate per epoch metrics
        avg_epoch_train_loss = epoch_accumulated_loss / len(trainloader)
        epoch_train_accuracy = epoch_train_correct / epoch_total

        # print(f"Epoch train loss {avg_epoch_train_loss}, acc {epoch_train_accuracy}, validatio loss {avg_val_loss}, acc: {val_accuracy}")
            


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