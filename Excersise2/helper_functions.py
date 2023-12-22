import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm.notebook import tqdm


def plot_training_progress(epochs, t_acc, v_acc, t_loss, v_loss) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14,4))


    ax[0].plot(range(1, epochs+1), t_acc, label='Ευστοχία εκπαίδευσης')
    ax[0].plot(range(1, epochs+1), v_acc, '--', label='Ευστοχία επικύρωσης')
    ax[0].set_xticks(range(1, epochs+1))
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xlabel('Εποχή')
    ax[0].set_ylabel('Ευστοχία')
    ax[0].set_title('Ευστοχία')
    ax[0].legend()

    ax[1].plot(range(1, epochs+1), t_loss, 'g-',label='Απώλεια εκπαίδευσης')
    ax[1].plot(range(1, epochs+1), v_loss, 'r--', label='Απώλεια επικύρωσης')
    ax[1].set_xticks(range(1, epochs+1))
    # ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel('Εποχή')
    ax[1].set_ylabel('Απώλεια')
    ax[1].set_title('Απώλεια')
    ax[1].legend()

    plt.suptitle('Εξέλιξη εκπαίδευσης')
    plt.show()

    
def display_conf_matrix(y_preds, test, classes):
    preds = np.array([x.tolist() for x in y_preds])
    if len(test[0]) == 3:
        y_train = [x[2] for x in test]
        
    else:
        y_train = [x[1] for x in test]

    # print(preds)
    # return preds

    conf_matrix = confusion_matrix(y_train, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes.values())
    disp.plot()
    plt.title('Πίνακας σύγχησης')
    plt.show()


def print_elapsed_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.
    """
    total_time = end - start
    print(f'Train time on {device}: {total_time:.3f} seconds')
    # return total_time


def train_net(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, 
          epochs: int = 10, optimizer: optim = None, loss: nn.modules.loss = None,
          device: str = 'cpu', print_period: int = 10) -> tuple:
    
    print(f'Training on {device}')
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    model.train()

    period_accumulated_loss = 0.0
    period_train_correct = 0
    period_total = 0

    for epoch in tqdm(range(epochs), desc='Epochs completed'):
        # Per epoch
        epoch_accumulated_loss = 0.0
        epoch_train_correct = 0
        epoch_total = 0

        # Per period
        # for batch, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc='Batches'):
        for batch, data in enumerate(trainloader, 1):
            reporting_step = epoch * len(trainloader) + batch

            if len(data) == 3:
                X, p, y = data
                X, p, y = X.to(device), p.to(device), y.to(device)
                pred = model(X, p)
            else:
                X, y = data
                X, y = X.to(device), y.to(device)
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
            
            if reporting_step % print_period == 0 and reporting_step != 0 :

                v_batch, val_running_loss, val_correct = vallidation(model, valloader, loss, device)            
                      
                #Calculate and print period metrics
                avg_period_train_loss = period_accumulated_loss / print_period      # Accumulated loss / number of batches of reporting period
                train_period_accuracy = period_train_correct / period_total         # Accumulated acc / number of samples in the training batches of reporting period
                avg_val_loss = val_running_loss / (v_batch + 1)                     # Accumulated loss / number of validation batches
                val_accuracy = val_correct / len(valloader.dataset)                 # Accumulated acc / number of samples in validation dataset (all batches)

                print(f"[Epoch: {epoch+1}, batch: {batch}] Train loss: {avg_period_train_loss:.3f}, Train acc: {train_period_accuracy:.3f} | Validation loss: {avg_val_loss:.3f}, Validation acc: {val_accuracy:.3f}")

                # Zero period counters
                period_accumulated_loss = 0.0
                period_train_correct = 0
                period_total = 0

        # Calculate epoch metrics
        v_batch, val_running_loss, val_correct = vallidation(model, valloader, loss, device)
        avg_val_loss = val_running_loss / (v_batch + 1)                     # Accumulated loss / number of validation batches
        val_accuracy = val_correct / len(valloader.dataset)                 # Accumulated acc / number of samples in validation dataset (all batches)


        # epochs_report = []
        train_loss_history.append(epoch_accumulated_loss / len(trainloader))
        train_acc_history.append(epoch_train_correct / epoch_total)
        val_loss_history.append(val_running_loss / (v_batch + 1))
        val_acc_history.append(val_correct / len(valloader.dataset))

        print(f"---Epoch {epoch+1} report | Train loss {train_loss_history[-1]:.3f}, Train accuracy {train_acc_history[-1]:.3f} | Validation loss {val_loss_history[-1]:.3f}, Validation accuracy: {val_acc_history[-1]:.3f}")

    return epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def vallidation(model: nn.Module, valloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> tuple:
    model.eval()
    with torch.inference_mode():
        val_running_loss = 0.0
        val_correct = 0
        # for v_batch, (X, y) in tqdm(enumerate(valloader, 0), total=len(valloader), desc='Validation'):
        for v_batch, data in enumerate(valloader, 0):
            if len(data) == 3:
                X, p, y = data
                X, p, y = X.to(device), p.to(device), y.to(device)
                pred_val = model(X, p)
            else:
                X, y = data
                X, y = X.to(device), y.to(device)
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


def test_net(model: nn.Module, testloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> list:
    total = 0
    correct = 0
    preds = []

    model.eval()
    with torch.inference_mode():
        test_running_loss = 0.0
        for data in testloader:        
            if len(data) == 3:
                X, p, y = data
                X, p, y = X.to(device), p.to(device), y.to(device)
                pred = model(X, p)
            else:
                X, y = data
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