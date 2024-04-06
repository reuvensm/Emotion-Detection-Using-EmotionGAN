import os
import pickle
import argparse

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# lightly
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import update_momentum
# optuna
import optuna

# FER2013 and more
from core_code.datasets.fer2013 import FER2013
from core_code.models.emotion_net import EmotionNet
from core_code.models.byol_model import BYOL
from core_code.core_utils.globals import device
from core_code.core_utils.transforms import train_transformations, test_transformations

OPTUNA_DIR = "optuna_study_output"

train_data = FER2013(phase='train', emotion_transform_prob=0, transform=train_transformations)
validation_data = FER2013(phase='val', emotion_transform_prob=0, transform=test_transformations)
train_data_byol = FER2013(phase='train', emotion_transform_prob=1.0, transform=test_transformations, byol=True)

def get_fer2013(batch_size):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader


def define_model(model_name, exp_name):
    model = EmotionNet(name=model_name, experiment_name=exp_name, fine_tune=True)
    return model

def define_model_byol(model_name, exp_name):
    net = EmotionNet(name=model_name, experiment_name=exp_name, fine_tune=True)
    backbone = nn.Sequential(*list(net.nn_model.children())[:-1])
    model = BYOL(backbone=backbone, net=model_name, fine_tune=True)
    return model

def objective_byol(trial, model_name, exp_name):
    # Generate the model.
    model = define_model_byol(model_name, exp_name).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # log=True, will use log scale to interplolate between lr
    # optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"])
    optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "SGD"])
    weight_decay = trial.suggest_float('weight_decay', 0 , 1e-4)
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_name = trial.suggest_categorical('scheduler', ["StepLR", "CosineAnnealingLR"])
    scheduler = StepLR(optimizer, 10, 0.1) if scheduler_name == "StepLR" else CosineAnnealingLR(optimizer, 45)
    criterion = NegativeCosineSimilarity()
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Get dataset
    train_loader_byol = torch.utils.data.DataLoader(train_data_byol, batch_size=batch_size, shuffle=True)

    epochs = 45
    n_train_examples = 128 * 30


    # Training of the model.
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader_byol):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            x1, _, x0  = data # x0 is the original, x1 is the augmented
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                              
        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(total_loss/ ((batch_idx+1) * batch_size), epoch)  
        
        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        scheduler.step()

    return total_loss/ ((batch_idx+1) * batch_size)

def objective(trial, model_name, exp_name):
    # Generate the model.
    model = define_model(model_name, exp_name).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # log=True, will use log scale to interplolate between lr
    # optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "SGD"])
    optimizer_name = trial.suggest_categorical('optimizer', ["Adam", "SGD"])
    weight_decay = trial.suggest_float('weight_decay', 0 , 1e-4)
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_name = trial.suggest_categorical('scheduler', ["StepLR", "CosineAnnealingLR"])
    scheduler = StepLR(optimizer, 10, 0.1) if scheduler_name == "StepLR" else CosineAnnealingLR(optimizer, 45)
    criterion = nn.CrossEntropyLoss()
    if model_name == "resnet18":
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    else: # evoiding large models GPU memory problems
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
    
    # Get dataset
    train_loader, valid_loader = get_fer2013(batch_size)

    epochs = 45
    n_train_examples = 128 * 30
    n_valid_examples = 128 * 10


    # Training of the model.
    for epoch in range(epochs):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * batch_size >= n_train_examples:
                break

            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)  # forward pass

            loss = criterion(outputs, labels)  # calculate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * batch_size >= n_valid_examples:
                    break
                inputs, labels, _ = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(accuracy, epoch)  
        
        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        scheduler.step()

    return accuracy


    
def run_optuna(model_name,exp_name, byol=False):
     # now we can run the experiment
    sampler = optuna.samplers.TPESampler()
    study_name = f"{model_name}_{exp_name}"
    if byol:
        study = optuna.create_study(
            study_name=study_name, direction='minimize', sampler=sampler,
        )
        study.optimize(
            lambda trial: objective_byol(trial, model_name=model_name, exp_name=exp_name),
            n_trials=20, timeout=4800
        )
    else:
        study = optuna.create_study(
            study_name=study_name, direction='maximize', sampler=sampler,
        )
        study.optimize(
            lambda trial: objective(trial, model_name=model_name, exp_name=exp_name),
            n_trials=30, timeout=4800
        )

    # Save the Optuna study object using pickle
    with open(f'{OPTUNA_DIR}/optuna_study_results_{study_name}.pkl', 'wb') as f:
        pickle.dump(study, f)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params

def plot_optuna_graphs(study_pkl_path):
    if not os.path.isfile(study_pkl_path):
        raise FileNotFoundError(study_pkl_path)
    
    with open(study_pkl_path, 'rb') as f:
        study = pickle.load(f)
    # Print the best trial
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save plots
    fig=optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{OPTUNA_DIR}/param_importances_{study.study_name}.png")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{OPTUNA_DIR}/optimization_history_{study.study_name}.png")


if __name__ == "__main__":
    if not os.path.isdir(OPTUNA_DIR):
        os.mkdir(OPTUNA_DIR)
    parser = argparse.ArgumentParser(description='Optuna HyperParams Study')
    parser.add_argument('--net_name', type=str, help='[resnet50]')
    parser.add_argument('--plot_results', default='', type=str, help='path_to_study_pkl')
    parser.add_argument('--exp_name', default='Regular', type=str, help='[Augmented, Regular]')
    parser.add_argument('--byol', default=False, type=bool, help='[True, False]')

    args = parser.parse_args()
    if args.plot_results == '':
        run_optuna(args.net_name, args.exp_name, byol=args.byol)
    else:
        plot_optuna_graphs(args.plot_results)
