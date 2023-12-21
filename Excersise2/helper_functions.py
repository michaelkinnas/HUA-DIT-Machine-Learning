import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def train_net(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, 
          epochs: int = 10, optimizer: optim = None, loss: nn.modules.loss = None,
          device: str = 'cpu', print_period: int = 10) -> tuple:
    
    print(f'Training on {device}')
    
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

                v_batch, val_running_loss, val_correct = vallidation(model, valloader, loss, device)            
                      
                #Calculate and print period metrics
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
                # model.train()


        # Calculate epoch metrics
        v_batch, val_running_loss, val_correct = vallidation(model, valloader, loss, device)
        avg_val_loss = val_running_loss / (v_batch + 1)                     # Accumulated loss / number of validation batches
        val_accuracy = val_correct / len(valloader.dataset)                 # Accumulated acc / number of samples in validation dataset (all batches)


        # epochs_report = []
        train_loss_history.append(epoch_accumulated_loss / len(trainloader))
        train_acc_history.append(epoch_train_correct / epoch_total)
        val_loss_history.append(val_running_loss / (v_batch + 1))
        val_acc_history.append(val_correct / len(valloader.dataset))

        print(f"Epoch {epoch} | Train loss {train_loss_history[-1]:.3f}, Train accuracy {train_acc_history[-1]:.3f} | Validation loss {val_loss_history[-1]:.3f}, Validation accuracy: {val_acc_history[-1]:.3f}")

    return epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def test_net(model: nn.Module, testloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> list:
    model.eval()
    total = 0
    correct = 0
    preds = []
    with torch.inference_mode():
        test_running_loss = 0.0
        for (X, y) in testloader:        
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # Calculate loss
            test_running_loss += loss(pred, y).item() 

            # Convert to predictions and calculate accuracy
            yhat = torch.argmax(pred, 1)

            for pred in yhat:
                preds.append(pred)
            
            total += y.size(0)
            correct += (yhat == y).type(torch.float).sum().item()

    print(f"Average loss: {test_running_loss/len(testloader.dataset):.4f}. Test accuracy in {total} images: {correct/total:.4f}")

    return preds


def vallidation(model: nn.Module, valloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> tuple:
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
    
    # Put model back into train mode
    model.train()    
    return v_batch, val_running_loss, val_correct


def plot_training_progress(epochs, t_acc, v_acc, t_loss, v_loss) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14,4))


    ax[0].plot(range(1, epochs+1), t_acc, label='Train accuracy')
    ax[0].plot(range(1, epochs+1), v_acc, '--', label='Validation accuracy')
    ax[0].set_xticks(range(1, epochs+1))
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].legend()

    ax[1].plot(range(1, epochs+1), t_loss, 'g-',label='Train loss')
    ax[1].plot(range(1, epochs+1), v_loss, 'r--', label='Validation loss')
    ax[1].set_xticks(range(1, epochs+1))
    # ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss')
    ax[1].legend()

    plt.suptitle('Training progress')
    plt.show()


# Test output of model
# test = torch.rand([1, 3, 100, 150], dtype=torch.float32, device=device)
# with torch.inference_mode():
#     model.eval()
#     output = model(test)
# print(f'Input shape: {test.shape}, output shape: {output.shape}')
    
def display_conf_matrix(y_preds, test, classes):
    preds = np.array([x.tolist() for x in y_preds])
    y_train = [x[1] for x in test]

    conf_matrix = confusion_matrix(y_train, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes.values())
    disp.plot()
    plt.title('Πίνακας σύγχησης')
    plt.show()


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.
    """
    total_time = end - start
    print(f'Train time on {device}: {total_time:.3f} seconds')
    # return total_time