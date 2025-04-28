import torch
import torch.nn as nn
import torch.nn.functional as F
import checkersBoard

# Define the Residual Block as a separate module
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Add the residual connection
        out = self.relu2(out)
        return out

class ResNN(nn.Module):
    def __init__(self, residual_blocks=2, width=64, q_learning_only=True):
        super(ResNN, self).__init__()
        self.board_height = checkersBoard.CheckersBoard.board_height
        self.board_width = checkersBoard.CheckersBoard.board_width
        self.action_size = checkersBoard.CheckersBoard.action_size
        self.q_learning_only = q_learning_only

        # Input channels correspond to the 5 planes from extract_features (assuming channels_first now)
        in_channels = 5

        # Initial convolutional block
        self.conv_in = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(width)
        self.relu_in = nn.ReLU()

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(width) for _ in range(residual_blocks)]
        )

        # Value head
        self.conv_val = nn.Conv2d(width, 32, kernel_size=1, bias=False) # 1x1 conv
        self.bn_val = nn.BatchNorm2d(32)
        self.relu_val = nn.ReLU()
        self.flatten_val = nn.Flatten()
        # Calculate flattened size: 32 channels * height * width
        fc1_in_features = 32 * self.board_height * self.board_width
        self.fc1_val = nn.Linear(fc1_in_features, width)
        self.relu_fc1_val = nn.ReLU()
        self.fc2_val = nn.Linear(width, 1)
        self.tanh_val = nn.Tanh()

        # Policy head (optional)
        if not q_learning_only:
            self.conv_pol = nn.Conv2d(width, 32, kernel_size=1, bias=False) # 1x1 conv
            self.bn_pol = nn.BatchNorm2d(32)
            self.relu_pol = nn.ReLU()
            self.flatten_pol = nn.Flatten()
            # Calculate flattened size: 32 channels * height * width
            fc_pol_in_features = 32 * self.board_height * self.board_width
            self.fc_pol = nn.Linear(fc_pol_in_features, self.action_size)
            # Softmax will be applied during loss calculation (CrossEntropyLoss) or manually if needed

    def forward(self, x):
        # Input x shape: (batch, channels, height, width)

        # Initial conv
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu_in(out)

        # Residual blocks
        out = self.res_tower(out)

        # Value head calculation
        val = self.conv_val(out)
        val = self.bn_val(val)
        val = self.relu_val(val)
        val = self.flatten_val(val)
        val = self.fc1_val(val)
        val = self.relu_fc1_val(val)
        value_out = self.fc2_val(val)
        value_out = self.tanh_val(value_out)

        if self.q_learning_only:
            return value_out
        else:
            # Policy head calculation
            pol = self.conv_pol(out)
            pol = self.bn_pol(pol)
            pol = self.relu_pol(pol)
            pol = self.flatten_pol(pol)
            policy_out = self.fc_pol(pol)
            # Note: Softmax is not applied here, CrossEntropyLoss expects raw logits
            return value_out, policy_out

# Removed Keras-specific methods: compile, fit_model, predict, load_weights, save_weights, save_model
# Removed set_lr for now, optimizer LR is handled externally
# Removed model.summary(), use print(model) in PyTorch
