import argparse
import random

import torch
import torch.nn as nn

from core_code.datasets.fer2013 import FER2013
from core_code.models.emotion_net import EmotionNet
from core_code.models.byol_model import BYOL
from core_code.core_utils.globals import device
from core_code.core_utils.transforms import test_transformations
from core_code.train_byol import train_byol
from train_model import get_resnet18_hyperparameters, get_densenet201_hyperparameters
from core_code.core_utils.logger_config import logger
from lightly.loss import NegativeCosineSimilarity

def get_args():
    parser = argparse.ArgumentParser(description='Emotion Detection Trainer')
    parser.add_argument('--net_name', type=str, help='[resnet18]')
    parser.add_argument('--exp_name', type=str, help='[AugmentedEmotion, Regular]') 
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--fine_tune', default=False, type=bool)
    return parser.parse_args()

# Train the model
def main():
    logger.info(f"\n#########\nBYOL Training!!!\n#########")
    random.seed(1)
    params = get_args()
    logger.info("Args parsed succesfully")
    logger.info("Params: " + str(params))
    
    net = EmotionNet(name=params.net_name, experiment_name=params.exp_name, fine_tune=params.fine_tune).to(device)
    backbone = nn.Sequential(*list(net.nn_model.children())[:-1])
    model = BYOL(backbone=backbone, net=params.net_name, fine_tune=params.fine_tune).to(device)
    
    logger.info(f"Model for BYOL was initialized and sent to {device}")
    logger.info(str(model))

    train_data = FER2013(phase='train', emotion_transform_prob=1.0, transform=test_transformations, byol=True)

    logger.info("All data has been loaded")

        
    if params.net_name == "resnet18":
        batch_size, optimizer, _ = get_resnet18_hyperparameters(model, epochs=params.epochs)
    elif params.net_name == "densenet201":
        batch_size, optimizer, _ = get_densenet201_hyperparameters(model, epochs=params.epochs)
    else:
        raise(NameError(f"Unkown net {params.net_name}"))
    
    # For effiency in BYOL
    batch_size = 32
    logger.info("Got hyper parametrs for current model:")
    logger.info("batch size=" + str(batch_size))
    logger.info(optimizer)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    logger.info("DataLoader initialized")
    
    criterion = NegativeCosineSimilarity()
    
    
    logger.info("HyperParams initiliazed, starting training...")
    train_byol(model=model, epochs=params.epochs, device=device, dataloader=train_loader,\
        criterion=criterion, optimizer=optimizer,net_name=params.net_name, exp_name=params.exp_name)
    logger.info("Train BYOL Ended!")
    
if __name__ == '__main__':
    main()
