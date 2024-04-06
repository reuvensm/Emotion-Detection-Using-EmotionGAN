import os
import numpy as np
import glob
import cv2
from core_code.core_utils.globals import NUM_OF_CLASSES
from core_code.datasets.fer2013 import FER2013
import pickle

class FER2013Exp(FER2013):
    
    def __init__(self, root='./data/fer2013', phase='train', transform=None,
                 target_transform=None, expanded_path='./data/expanded_fer_2013'):
        super().__init__(root, phase, transform, target_transform, emotion_transform_prob = 0)
        self._expanded_path = expanded_path
        
        if (os.path.isfile(os.path.join(root, 'processed', 'train_expanded.pkl'))):
            if self._phase == 'train':
                self._train_data, self._train_labels = pickle.load(
                    open(os.path.join(self._root, 'processed', 'train_expanded.pkl'), 'rb'))
                print("Expanded Dataset already processed")
        else:
            self.expanded_process()
    
    def expanded_process(self):
        if self._phase == 'train':
            new_images_arrays = []
            new_labels_arrays = []
            for i in range(NUM_OF_CLASSES):
                new_images = glob.glob(os.path.join(self._expanded_path, f'{i}/*.png'))
                new_image_array = np.zeros(shape=(len(new_images), 48, 48)).astype(np.uint8)
                for j, image_path in enumerate(new_images):
                    image =  cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    image = np.reshape(image, (48, 48))
                    new_image_array[j] = image
                new_image_array = new_image_array.reshape(len(new_images), 48, 48, 1)
                new_image_array = np.concatenate((new_image_array, new_image_array, new_image_array), axis=3)
                new_images_arrays.append(new_image_array)
                new_label_array = np.ones((len(new_images),)) * i
                new_labels_arrays.append(new_label_array)
            self._train_data = np.vstack((self._train_data, *new_images_arrays)).astype(np.uint8)
            self._train_labels = np.hstack((self._train_labels, *new_labels_arrays)).astype(np.uint8)
            self.TrainSize = len(self._train_labels)
            pickle.dump(
            (self._train_data, self._train_labels),
            open(os.path.join(self._root, 'processed', f'train_expanded.pkl'), 'wb'))
