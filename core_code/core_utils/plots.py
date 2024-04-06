
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from core_code.core_utils.globals import *
matplotlib.style.use('ggplot')

PLOTS_DIR = 'plots_output'

if not os.path.isdir(PLOTS_DIR):
    os.mkdir(PLOTS_DIR)

# Plot confusion matrix:
def confusion_matrix(model, confusion_matrix, accuracy, test_name='Test', to_save=False):
    """
    plot and save model confusion matrix, usually on test set
    """
    print("Accuracy: {:.3f}%".format(accuracy))
    classes = np.array(['angry', 'disgust', 'fear', 'happy', 'netural', 'sad', 'surprise'])
    
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    matshow = ax.matshow(cm_normalized, aspect='auto', vmin=0, vmax=np.max(cm_normalized), cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(len(classes)), classes)
    plt.grid(visible=None)
    
    # Add colorbar
    fig.colorbar(matshow, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, str(round(cm_normalized[i, j],2)), ha='center', va='center',
                    color='white' if cm_normalized[i, j] > np.max(cm_normalized) / 2 else 'black')
    plt.title(f'{test_name} Accuracy: {round(accuracy, 2)}%')
    if to_save:
        plt.savefig(f"{PLOTS_DIR}/confusion_matrix_{model.name}_{model.experiment_name}_{test_name}.png")


def plot_train_validation_accuracy(model, train_acc, valid_acc, same_figure=False):
    """
    Plot loss and accuracy of 
    """
    if same_figure:
        training_epochs = np.arange(5, len(train_acc) * 5 + 1, 5)
        validation_epochs = np.arange(1, len(valid_acc) + 1)

        plt.figure(figsize=(10, 7))
        plt.ylim(40, 100)
        plt.plot(training_epochs, train_acc, '-o', label='Training Accuracy', color='blue')
        plt.plot(validation_epochs, valid_acc, '-o', label='Validation Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        #axes = plt.axes()
        #axes.set_ylim([0, 1])
        #plt.xticks(np.arange(1, max(len(train_acc) * 5, len(valid_acc)) + 1, 1))
        #plt.yticks(np.linspace(0, 1, 11))
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{PLOTS_DIR}/train_validation_accuracy_{model.name}_{model.experiment_name}.png")
        return
    
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(range(5, len(train_acc)*5+1, 5),train_acc, color='blue', linestyle='-', label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/train_accuracy_{model.name}_{model.experiment_name}.png")
    
    plt.figure(figsize=(10, 7))
    plt.plot(range(1,len(valid_acc)+1),valid_acc, color='red', linestyle='-', label='Validataion Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/validation_accuracy_{model.name}_{model.experiment_name}.png")
