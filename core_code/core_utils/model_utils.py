
import torch
import torch.nn as nn
import os

from core_code.core_utils.globals import *
from core_code.models.emotion_net import EmotionNet
from core_code.models.byol_model import BYOL

def save_model(model, optimizer, mean_loss, epoch, filename, checkpoint_dir='checkpoints', byol=False):
    # print('==> Saving model ...')
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'loss': mean_loss,
        'name': None,
        'experiment': None
    }
    if not byol:
        state['name'] = model.name
        state['experiment'] = model.experiment_name
        
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))


def load_model(checkpoint_dir='checkpoints', name='resnet50', exp_name='', byol=False):
    if byol:
        net = EmotionNet(name=name)
        backbone = nn.Sequential(*list(net.nn_model.children())[:-1])
        model = BYOL(backbone=backbone, net=name)
        filename = os.path.join(checkpoint_dir, f'ckpt_byol_{name}_{exp_name}_last_trained_epoch.pth')
    else:
        model = EmotionNet(name=name)
        filename = os.path.join(checkpoint_dir, f'ckpt_{name}_{exp_name}_best_validation.pth')
    model.load_state_dict(torch.load(filename)['net'])
    return model
