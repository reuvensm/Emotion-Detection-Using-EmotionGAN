import argparse
from core_code.datasets.augmentations_maker import create_emotion_augmentations, expand_train_dataset
from core_code.datasets.fer2013 import FER2013
from core_code.core_utils.transforms import test_transformations

def get_train_data_fer2013():
    # Use emotion_transform_prob as we want to get the original images
    # Use test_transformations as we don't want to apply augmentations
    train_data = FER2013(phase='train', emotion_transform_prob=0, transform=test_transformations)
    return train_data


def main():
    parser = argparse.ArgumentParser(description='Emotion Augmentations Creator')
    parser.add_argument('--dataset', default='fer2013', type=str, help='[fer2013]')
    parser.add_argument('--dest_folder', default='data/fer2013_augmentations', type=str)
    parser.add_argument('--num_augments', default=10, type=int, help='How many Augments to generate from each image')
    parser.add_argument('--expand_dataset', default=False, type=bool, help='If true will expand fer2013')
    parser.add_argument('--folder_batch_size', default=100, type=int, help='num of images in each index')
    args = parser.parse_args()

    train_data = None
    if args.dataset == 'fer2013':
        train_data = get_train_data_fer2013()
    else:
        raise NameError(args.dataset)
    
    if args.expand_dataset:
        expand_train_dataset(train_data=train_data,
                                    dest_folder=args.dest_folder)
    else:
        create_emotion_augmentations(train_data=train_data,
                                    dest_folder=args.dest_folder,
                                    num_augmentations=args.num_augments,
                                    folder_batch_size=args.folder_batch_size)

if __name__ == '__main__':
    main()
