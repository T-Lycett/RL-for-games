import torch
import torch.nn as nn
import checkersBoard
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.lr = lr # Learning rate is handled by the optimizer externally
        board_height = checkersBoard.CheckersBoard.board_height # Should be 8
        board_width = checkersBoard.CheckersBoard.board_width # Should be 8
        in_channels = 5 # Input planes

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Final 1x1 convolution reducing channels to 1
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu4 = nn.ReLU()

        self.flatten = nn.Flatten()
        # Calculate flattened size: 1 channel * height * width
        fc1_in_features = 1 * board_height * board_width
        self.fc1 = nn.Linear(fc1_in_features, 128)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 1)
        self.tanh_out = nn.Tanh()

        # Keras model.summary() equivalent is printing the model instance
        # print(self)

    def forward(self, x):
        # Input x shape: (batch, channels, height, width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)

        x = self.fc2(x)
        x = self.tanh_out(x)

        return x

# Removed Keras-specific methods: compile, set_lr, fit_model, predict, load_weights, save_weights, save_model, load_model
