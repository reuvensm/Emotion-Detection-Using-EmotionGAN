
import torch
import time
import numpy as np
from typing import List, Tuple

from core_code.core_utils.globals import *
from core_code.core_utils.model_utils import save_model
from core_code.core_utils.logger_config import logger


# function to calcualte accuracy of the model
def calculate_accuracy(model, device, dataloader, criterion, create_confusion_matrix=True):
    model.eval() # put in evaluation mode,  turn off Dropout, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    running_loss = 0.0
    confusion_matrix = np.zeros([NUM_OF_CLASSES, NUM_OF_CLASSES], int)
    with torch.no_grad():
        for data in dataloader:
            # Evaluate model on data
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy & confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            if create_confusion_matrix:
                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), predicted[i].item()] += 1
            
            
    total_loss = running_loss / len(dataloader.dataset)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix, total_loss


def train_checkpoint(model, epoch, optimizer, loss, epoch_start_time, device, trainloader, validation_loader, criterion, only_val_checkpoint=False):
    if only_val_checkpoint:
        train_accuracy = None
    else:
        train_accuracy, _, _ = calculate_accuracy(model, device, trainloader, criterion, create_confusion_matrix=False)
    validation_accuracy, _, _ = calculate_accuracy(model, device, validation_loader,criterion, create_confusion_matrix=False)
    epoch_time = time.time() - epoch_start_time
    if only_val_checkpoint:
        log_line = "Epoch: {} | Loss: {:.4f} | Validation accuracy: {:.3f}% | Epoch Time: {:.2f} secs".format(epoch, loss, validation_accuracy, epoch_time)
    else:
        log_line = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Validation accuracy: {:.3f}% | Epoch Time: {:.2f} secs".format(epoch, loss, train_accuracy, validation_accuracy, epoch_time)
    print(log_line)
    logger.info(log_line)
    save_model(model, optimizer, loss, epoch, f'ckpt_{model.name}_{model.experiment_name}_last_trained_epoch.pth')  # Save in case of failure
    return train_accuracy, validation_accuracy


def train(model, epochs, device, trainloader, validation_loader, criterion, optimizer, scheduler, verbose=False) -> Tuple[List[float], List[float]]:
    # training loop
    train_accuracy_list, validation_accuracy_list = [], []
    best_validation_accuracy = 0
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode, turn on Dropout, BatchNorm uses batch's statistics
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, _ = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Kornia:
            # inputs = aug_list(inputs)
            
            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()
            
            if verbose and i % 1 == 0:
                log_line = f"\t In epoch number {epoch}, finished iteration {i} | Running loss: {running_loss/ len(trainloader.dataset):.3f} | Time from start of this epoch: {time.time()- epoch_time:.2f} secs"
                logger.info(log_line)

        # Normalizing the loss by the total number of train batches
        mean_loss = running_loss / len(trainloader.dataset) 
        # Scheduler
        scheduler.step()
        
        # save model checkpoint
        if epoch % 5 == 0:
            # Calculate training/test set accuracy of the existing model
            train_accuracy, validation_accuracy = train_checkpoint(model, epoch, optimizer, mean_loss, epoch_time, device, trainloader, validation_loader, criterion)
            train_accuracy_list.append(train_accuracy)
            validation_accuracy_list.append(validation_accuracy)
        else:
            # Calculate training/test set accuracy of the existing model
            _, validation_accuracy = train_checkpoint(model, epoch, optimizer, mean_loss, epoch_time, device, trainloader, validation_loader, criterion, only_val_checkpoint=True)
            validation_accuracy_list.append(validation_accuracy)
        
        # else:
        #     log_line = f"Finished epoch number {epoch} | Loss: {mean_loss} | Epoch Time: {time.time()- epoch_time}"
        #     logger.info(log_line)
            
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_model(model, optimizer, mean_loss, epoch, f'ckpt_{model.name}_{model.experiment_name}_best_validation.pth')  # Save the best validation model
    
    # Save last checkpoint model - Note: this is not necessary the best validation model. 
    save_model(model, optimizer, mean_loss, epoch, f'ckpt_{model.name}_{model.experiment_name}_final_checkpoint.pth')   # Save the end-of-run model
    return train_accuracy_list, validation_accuracy_list
