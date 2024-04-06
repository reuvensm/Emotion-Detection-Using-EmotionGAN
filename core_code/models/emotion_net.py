
import torch
import torch.hub
import torch.nn as nn
from torchvision import models

from core_code.core_utils.globals import NUM_OF_CLASSES

class EmotionNet(nn.Module):
    resnet_18_model = 'resnet18'
    densenet_201_model = 'densenet201'
    vgg_19_model = 'vgg19'
    resnet_34_model = 'resnet34'
    dino_v2_model = 'dino_v2'

    def __init__(self, name, num_classes=NUM_OF_CLASSES, pretrained=True, fine_tune=False, experiment_name="Regular"):
        super(EmotionNet, self).__init__()
        self.name = name
        self.experiment_name = experiment_name
        self.weights = 'DEFAULT' if pretrained else None
        # Load pretrained Models
        if name == self.resnet_18_model:
            self.nn_model = models.resnet18(weights=self.weights)
            self.set_grad(fine_tune=fine_tune)
            num_of_features = self.nn_model.fc.in_features
            self.nn_model.fc = nn.Linear(num_of_features, num_classes)
            
        elif name == self.resnet_34_model:
            self.nn_model = models.resnet34(weights=self.weights)
            self.set_grad(fine_tune=fine_tune)
            num_of_features = self.nn_model.fc.in_features
            self.nn_model.fc = nn.Linear(num_of_features, num_classes)

        elif name == self.vgg_19_model:
            self.nn_model = models.vgg19(weights=self.weights)
            self.set_grad(fine_tune=fine_tune)
            num_of_features = self.nn_model.classifier[6].in_features
            self.nn_model.classifier[6] = nn.Linear(num_of_features, num_classes)
            
        elif name == self.densenet_201_model:
            self.nn_model = models.densenet201(weights=self.weights)
            self.set_grad(fine_tune=fine_tune)
            num_of_features = self.nn_model.classifier.in_features
            self.nn_model.classifier = nn.Linear(num_of_features, num_classes)

        elif name == self.dino_v2_model:
            self.nn_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            # self.nn_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.set_grad(fine_tune=fine_tune)  # Please never use fine tune here!!
            # num_of_features = 384
            num_of_features = 1024
            self.nn_model.head = nn.Linear(num_of_features, num_classes)
            self.show_requires_grad(fine_tune)
        else:
            raise NameError('No Model Found')

    def forward(self, images):
        return self.nn_model(images)
    
    def set_grad(self, fine_tune: bool) -> None:
        for param in self.nn_model.parameters():
            param.requires_grad = fine_tune
            
    def show_requires_grad(self, fine_tune):
        if not fine_tune:
            for name, param in self.nn_model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
