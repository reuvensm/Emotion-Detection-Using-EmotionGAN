import os
import numpy as np
from tqdm import tqdm
import torch
from cv2 import imwrite
from torch.utils.data import Dataset
from core_code.core_utils.emotionGAN_utils import get_augmentation_by_emotion
from core_code.core_utils.globals import NUM_OF_CLASSES
from EmotionGAN.utils.notebook_utils import GANmut

NUM_TRYS = 3

def create_emotion_augmentations(train_data: Dataset, dest_folder: str, num_augmentations: int, folder_batch_size=100) -> None:
    """
    Create augmentations for images from train loader
    Augmentations will be saved in the 'dest_folder' directory in the following format:
    <dest_folder>/<im_idx>_<aug_idx>.png 
    """
    # Create the destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

    # Initialize GAN model
    ganmut_model = GANmut(G_path='EmotionGAN/learned_generators/gaus_2d/1800000-G.ckpt', model='gaussian')
    # Generate and save augmentations
    for i, data in enumerate(tqdm(train_loader, desc="Creating Augmentations")):
        # get the inputs
        inputs, labels, _ = data
        # Save augmentations
        for j in range(inputs.size(0)):
            assert j < 1
            for k in range(num_augmentations):
                img = (inputs[j].permute((1,2,0)).numpy() * 255).astype(np.uint8)
                label = (labels[j].item())
                is_success = False
                for _ in range(NUM_TRYS):
                    augmnented_image, is_success = get_augmentation_by_emotion(img, label, ganmut_model, p=1.0)
                    if is_success:
                        break
                else: # GAN cannot augment this image
                    continue
                folder_num = i % folder_batch_size
                if not os.path.exists(os.path.join(dest_folder, str(folder_num))):
                    os.makedirs(os.path.join(dest_folder, str(folder_num)))
                imwrite(os.path.join(dest_folder, str(folder_num), f"{i}_{k}.png"), augmnented_image)

def expand_train_dataset(train_data: Dataset, dest_folder: str) -> None:
    emotions_probabilities = train_data.get_labels_probabilities()
    
    # Create the destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    # Initialize GAN model
    ganmut_model = GANmut(G_path='EmotionGAN/learned_generators/gaus_2d/1800000-G.ckpt', model='gaussian')
    # Generate and save augmentations
    for i, data in enumerate(tqdm(train_loader, desc="Creating New Dataset Images")):
        # get the inputs
        inputs, labels, _ = data
        # Save augmentations
        for j in range(inputs.size(0)):
            assert j < 1
            img = (inputs[j].permute((1,2,0)).numpy() * 255).astype(np.uint8)
            label = (labels[j].item())
            for k in range(NUM_OF_CLASSES):
                if label == k:
                    continue  # skip creating same emotion
                # Skip in emotions_probabilities[k] of the cases
                if torch.rand(1).item() < emotions_probabilities[k]:
                    continue 
                is_success = False
                for _ in range(NUM_TRYS):
                    augmnented_image, is_success = get_augmentation_by_emotion(img, k, ganmut_model, p=1.0, neighborhood=0.05)
                    if is_success:
                        break
                else: # GAN cannot augment this image
                    continue
                if not os.path.exists(os.path.join(dest_folder, str(k))):
                    os.makedirs(os.path.join(dest_folder, str(k)))
                imwrite(os.path.join(dest_folder, str(k), f"{i}.png"), augmnented_image)
