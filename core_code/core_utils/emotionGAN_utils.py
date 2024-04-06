
import torch
import numpy as np
from EmotionGAN.utils.notebook_utils import GANmut

def label2emotion():
    return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def map_emotion_to_coordinate(emotion):
    emotions_dict = label2emotion()
    emotion2coor = {'Angry': (0.6,-0.9),
                    'Disgust': (-0.7,-1),
                    'Fear': (-0.25,-0.9),
                    'Happy': (0.6,0.4),
                    'Sad': (-0.9,-0.6),
                    'Surprise': (0.9,-0.45),
                    'Neutral': (0.4, -0.5)}
    return emotion2coor[emotions_dict[emotion]]

def get_augmentation_by_emotion(image: np.ndarray, emotion: int, ganmut_model: GANmut, p=1.0, neighborhood=0.2):
    if torch.rand(1).item() > p:
        return image, False
    x_emotion, y_emotion = map_emotion_to_coordinate(emotion)
    min_x, min_y = x_emotion - neighborhood, y_emotion - neighborhood
    if emotion == 0:  # Angry has bigger sensitivity around the y axis
        min_y = y_emotion
    max_x, max_y = x_emotion + neighborhood, y_emotion + neighborhood
    min_x, min_y = max(min_x, -1),  max(min_y, -1)
    max_x, max_y = min(max_x, 1),  min(max_y, 1)
    generate_random_number = lambda a, b: torch.rand(1).item() * (b - a) + a
    rand_x, rand_y = generate_random_number(min_x, max_x), generate_random_number(min_y, max_y)
    result, is_success = ganmut_model.emotion_edit(image, x = rand_x, y = rand_y, save=False)
    return result, is_success
