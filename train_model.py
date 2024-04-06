
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from core_code.datasets.fer2013 import FER2013
from core_code.datasets.fer2013_expanded import FER2013Exp
from core_code.models.emotion_net import EmotionNet
from core_code.models.byol_model import BYOL_Fine_Tune
from core_code.core_utils.globals import device
from core_code.core_utils.transforms import train_transformations, test_transformations
from core_code.core_utils.model_utils import load_model
from core_code.train import train


from core_code.core_utils.logger_config import logger
from core_code.core_utils.plots import plot_train_validation_accuracy


def get_args():
    parser = argparse.ArgumentParser(description='Emotion Detection Trainer')
    parser.add_argument('--net_name', type=str, help='[resnet50]')
    parser.add_argument('--exp_name', type=str, help='[AugmentedEmotion, Regular]') 
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--emotion_transform_prob', type=float)
    parser.add_argument('--use_expanded_dataset', default=False, type=bool, help='Use fer2013 expanded dataset with new generated emotions')
    parser.add_argument('--fine_tune', default=False, type=bool)
    parser.add_argument('--load_byol', default=False, type=bool)
    parser.add_argument('--verbose', default=False, type=bool)
    return parser.parse_args()

# From optuna optimiziations:
def get_resnet50_hyperparameters(model):
    lr = 2e-4
    batch_size = 64
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = StepLR(optimizer, 10, 0.5)
    return batch_size, optimizer, scheduler

# Based on optuna results
def get_resnet18_hyperparameters(model,epochs):
    weight_decay = 2e-5
    batch_size = 128
    lr = 2e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler =  CosineAnnealingLR(optimizer,epochs)
    return batch_size, optimizer, scheduler

def get_densenet201_hyperparameters(model,epochs):
    weight_decay = 3.4e-5
    batch_size = 32
    lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler =  CosineAnnealingLR(optimizer,epochs)
    return batch_size, optimizer, scheduler

def get_dino_v2_hyperparameters(model,epochs):
    return get_resnet18_hyperparameters(model, epochs)

# Train the model
def main():
    random.seed(1)
    params = get_args()
    logger.info("Args parsed succesfully")
    logger.info("Params: " + str(params))
    
    if params.load_byol:
        byol_model = load_model(name=params.net_name, exp_name=params.exp_name, byol=True)
        model = BYOL_Fine_Tune(byol_model.backbone, name=params.net_name, experiment_name=params.exp_name, fine_tune=params.fine_tune).to(device)
        logger.info(f"BYOL Model was loaded successfully and sent to {device}")
    else:
        model = EmotionNet(name=params.net_name, experiment_name=params.exp_name, fine_tune=params.fine_tune).to(device)
        logger.info(f"Model was initialized and sent to {device}")
    logger.info(str(model))
    
    if params.use_expanded_dataset:
        train_data = FER2013Exp(phase='train', transform=train_transformations)
    else:
        train_data = FER2013(phase='train', emotion_transform_prob=params.emotion_transform_prob, transform=train_transformations)
    validation_data = FER2013(phase='val', emotion_transform_prob=0, transform=test_transformations)
    logger.info("All data has been loaded")

    if params.net_name == "resnet50":
        batch_size, optimizer, scheduler = get_resnet50_hyperparameters(model)
        
    if params.net_name == "resnet18":
        batch_size, optimizer, scheduler = get_resnet18_hyperparameters(model, epochs=params.epochs)
    
    if params.net_name == 'densenet201':
        batch_size, optimizer, scheduler = get_densenet201_hyperparameters(model, epochs=params.epochs)
    
    if params.net_name == 'dino_v2':
        batch_size, optimizer, scheduler = get_resnet18_hyperparameters(model, epochs=params.epochs)
    
    
    logger.info("Got hyper parametrs for current model:")
    logger.info("batch size=" + str(batch_size))
    logger.info(optimizer)
    logger.info(scheduler)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    logger.info("DataLoaders initialized")

    

    criterion = nn.CrossEntropyLoss()
    
    
    logger.info("HyperParams initiliazed, starting training...")
    train_score_list, val_score_list = train(model, epochs=params.epochs, device=device, trainloader=train_loader, validation_loader=validation_loader,\
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, verbose=params.verbose)
    logger.info("Train Ended!")
    logger.info("Saving train statistics")
    plot_train_validation_accuracy(model=model, train_acc=train_score_list, valid_acc=val_score_list)
    logger.info("Save statistics done!")
    
if __name__ == '__main__':
    main()
