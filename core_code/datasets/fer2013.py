import glob
import torch
import pickle
import numpy as np
import pandas as pd
import os
import cv2
from core_code.core_utils.globals import *

from EmotionGAN.utils.notebook_utils import GANmut


class FER2013(torch.utils.data.Dataset):
    """FER2013 Dataset.

    Args:
        _root, str: Root directory of dataset.
        _phase ['train'], str: train/val/test.
        _transform [None], function: A transform for a PIL.Image
        _target_transform [None], function: A transform for a label.

        _train_data, np.ndarray of shape N*3*48*48.
        _train_labels, np.ndarray of shape N.
        _val_data, np.ndarray of shape N*3*48*48.
        _val_labels, np.ndarray of shape N.
        _test_data, np.ndarray of shape N*3*48*48.
        _test_labels, np.ndarray of shape N.
    """
    Phase2Usage = {'Training': 'train', 'PrivateTest': 'val', 'PublicTest': 'test'}
    TrainSize = 28709
    ValSize = 3589
    TestSize = 3589

    def __init__(self, root='./data/fer2013', phase='train', transform=None,
                 target_transform=None, emotion_transform_prob = 1.0, GANmut_path='EmotionGAN',
                 augments_path='./data/fer2013_augmentations', byol=False):
        self._root = os.path.expanduser(root)
        self._phase = phase
        self._transform = transform
        self._target_transform = target_transform
        self._emotion_transform_prob = emotion_transform_prob
        self._augments_path = augments_path
        self._byol = byol
        self.ganmut_model = GANmut(G_path=os.path.join(GANmut_path, 'learned_generators/gaus_2d/1800000-G.ckpt') ,model='gaussian')
        self.success_rate = {'train': 0, 'val': 0, 'test': 0}
        self.total_rate = {'train': 0, 'val': 0, 'test': 0}

        if (os.path.isfile(os.path.join(root, 'processed', 'train.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'val.pkl'))
            and os.path.isfile(os.path.join(root, 'processed', 'test.pkl'))):
            print('Dataset already processed.')
        else:
            self.process('Training', self.TrainSize)
            self.process('PrivateTest', self.ValSize)
            self.process('PublicTest', self.TestSize)

        if self._phase == 'train':
            self._train_data, self._train_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'train.pkl'), 'rb'))
        elif self._phase == 'val':
            self._val_data, self._val_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'val.pkl'), 'rb'))
        elif self._phase == 'test':
            self._test_data, self._test_labels = pickle.load(
                open(os.path.join(self._root, 'processed', 'test.pkl'), 'rb'))
        else:
            raise ValueError('phase should be train/val/test.')

    def __getitem__(self, index):
        """Fetch a particular example (X, y).

        Args:
            index, int.

        Returns:
            image, torch.Tensor.
            label, int.
        """
        if self._phase == 'train':
            image, label = self._train_data[index], self._train_labels[index]
        elif self._phase == 'val':
            image, label = self._val_data[index], self._val_labels[index]
        elif self._phase == 'test':
            image, label = self._test_data[index], self._test_labels[index]
        else:
            raise ValueError('phase should be train/val/test.')

        # image = PIL.Image.fromarray(image.astype('uint8'))
        image = image.astype('uint8')
        image_orig = image.copy()
        folder_num = index % 100
        if self._emotion_transform_prob == 0:
            is_success = False
        else:
            augments_path = glob.glob(os.path.join(self._augments_path, str(folder_num), f'{index}_*.png'))
            is_success = (len(augments_path) > 0) and (torch.rand(1).item() <= self._emotion_transform_prob)
        # image, is_success = get_augmentation_by_emotion(image, label, self.ganmut_model, p=self._emotion_transform_prob)
        if is_success:
            augment_path = augments_path[int(torch.randperm(len(augments_path))[0].item())]
            image = cv2.imread(augment_path).astype('uint8')
            self.success_rate[self._phase] += 1  # Update success rate of face augmentation
        self.total_rate[self._phase] += 1
        if self._transform is not None:
            image = self._transform(image)
            if self._byol:
                image_orig = self._transform(image_orig)
        if self._target_transform is not None:
            label = self._target_transform(label)

        return image, label, image_orig

    def __len__(self):
        """Dataset length.

        Returns:
            length, int.
        """
        if self._phase == 'train':
            return len(self._train_data)
        elif self._phase == 'val':
            return len(self._val_data)
        elif self._phase == 'test':
            return len(self._test_data)
        else:
            raise ValueError('phase should be train/val/test.')
    
    def get_face_augmentation_success_rate(self):
        """
        Success rate
        """
        if self.total_rate[self._phase] == 0:
            return 0.0
        return self.success_rate[self._phase] / self.total_rate[self._phase]
    
    def get_labels_probabilities(self):
        if self._phase == 'train':
            labels = self._train_labels
        elif self._phase == 'val':
            labels = self._val_labels
        elif self._phase == 'test':
            labels = self._test_labels
        
        unique_values, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return dict(zip(unique_values, probabilities))

    def process(self, phase, size):
        """Fetch train/val/test data from raw csv file and save them onto
        disk.

        Args:
            phase, str: 'train'/'val'/'test'.
            size, int. Size of the dataset.
        """
        if phase not in ['Training', 'PrivateTest', 'PublicTest']:
            raise ValueError('phase should be train/val/test')        
        # Load all data.
        data_frame = pd.read_csv(os.path.join(self._root, 'icml_face_data.csv'))
        data_frame = data_frame[data_frame[' Usage']==phase]
        # Fetch all labels.
        labels = np.array(list(map(int, data_frame['emotion'])))  # np.ndarray
        assert labels.shape == (size,)

        # Fetch all images.
        image_array = np.zeros(shape=(len(data_frame), 48, 48))
        for i, row in enumerate(data_frame.index):
            image = np.fromstring(data_frame.loc[row, ' pixels'], dtype=int, sep=' ')
            image = np.reshape(image, (48, 48))
            image_array[i] = image

        # images = data_frame.values.astype('float64')
        assert image_array.shape == (size, 48, 48)
        image_array = image_array.reshape(size, 48, 48, 1)

        image_array = np.concatenate((image_array, image_array, image_array), axis=3)
        assert image_array.shape == (size, 48, 48, 3)
        if not os.path.isdir(os.path.join(self._root, 'processed')):
            os.mkdir(os.path.join(self._root, 'processed'))
        pickle.dump(
            (image_array, labels),
            open(os.path.join(self._root, 'processed', f'{self.Phase2Usage[phase]}.pkl'), 'wb'))
