import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self, config):
        super(BasicCNN, self).__init__()
        model_config = config['model_config']['basic_cnn']
        
        self.conv1 = nn.Conv2d(3, model_config['conv1_out'], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(model_config['conv1_out'], model_config['conv2_out'], 5)
        self.fc1 = nn.Linear(model_config['conv2_out'] * 5 * 5, model_config['fc1_out'])
        self.fc2 = nn.Linear(model_config['fc1_out'], model_config['fc2_out'])
        self.fc3 = nn.Linear(model_config['fc2_out'], config['model_config']['num_classes'])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x