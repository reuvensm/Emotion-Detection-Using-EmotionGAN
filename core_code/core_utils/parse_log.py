
import re
from core_code.models.emotion_net import EmotionNet
from core_code.core_utils.plots import plot_train_validation_accuracy
def parse_log_file(log_file):
    validation_accuracies = []
    train_accuracies = []
    model_name = None
    experiment_name = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
        # Extract model name and experiment name
        params_line = lines[1]
        match = re.search(r"exp_name='(\w+)'", params_line)
        if match:
            experiment_name = match.group(1)
        match = re.search(r"net_name='(\w+)'", params_line)
        if match:
            model_name = match.group(1)
        
        # Extract validation and training accuracies
        for line in lines:
            if 'Validation accuracy' in line:
                match = re.search(r'Validation accuracy: (\d+\.\d+)%', line)
                if match:
                    validation_accuracies.append(float(match.group(1)))
            
            if 'Training accuracy' in line:
                match = re.search(r'Training accuracy: (\d+\.\d+)%', line)
                if match:
                    train_accuracies.append(float(match.group(1)))
    
    print(model_name, experiment_name, len(validation_accuracies), len(train_accuracies))
    assert model_name is not None and experiment_name is not None\
        and len(validation_accuracies) > 0 and len(train_accuracies) > 0
        
    return validation_accuracies, train_accuracies, model_name, experiment_name

def plot_from_log(log_file):
    validation_accuracies, train_accuracies, model_name, experiment_name = parse_log_file(log_file)
    model = EmotionNet(name=model_name, experiment_name=experiment_name)
    plot_train_validation_accuracy(model, train_accuracies, validation_accuracies, same_figure=True)

if __name__ == '__main__':
    plot_from_log('logs/training_2024-03-30-14-23-52.log')