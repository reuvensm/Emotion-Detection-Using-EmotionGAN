import torch.nn as nn

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
import copy

from core_code.core_utils.globals import NUM_OF_CLASSES

class BYOL(nn.Module):
    def __init__(self, backbone, net='resnet18', fine_tune=True):
        super().__init__()
        if net=='resnet18':
            self.last_layer_dim = 512
            
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(self.last_layer_dim, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        
        self.set_grad(fine_tune)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
    
    def set_grad(self, fine_tune: bool) -> None:
        for param in self.backbone.parameters():
                param.requires_grad = fine_tune

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class BYOL_Fine_Tune(nn.Module):
    def __init__(self, backbone, name, experiment_name, fine_tune=True):
        super(BYOL_Fine_Tune, self).__init__()
        self.name = name
        self.experiment_name = experiment_name
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_OF_CLASSES)
        )
        self.set_grad(fine_tune)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        out = self.classifier(x)
        return out
    
    def set_grad(self, fine_tune: bool) -> None:
        for param in self.backbone.parameters():
                param.requires_grad = fine_tune
        for param in self.classifier.parameters():
                param.requires_grad = fine_tune
