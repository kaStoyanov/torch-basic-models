import torch
import torch.nn as nn
import torch.nn.functional as F

from mask import MaskLayer

class DQN(nn.Module):

    def __init__(self):
        
        super(DQN, self).__init__()
        
        # input size = 8 (rows) x 8 (cols) x 16 (bitboards)
        # - 6 bitboards for white pieces
        # - 6 bitboards for black pieces
        # - 1 for empty squares
        # - 1 for castling rights
        # - 1 for en passant
        # - 1 for player
        
        # first convolutional layer 8x8x16 => 8x8x32
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # second convolutional layer 8x8x32 => 8x8x64 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # third convolutional layer 8x8x64 => 8x8x128 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # first fully connected layer 8192 => 8192
        self.fc1 = nn.Linear(128*64, 128*64)
        
        # second fully connected layer 8192 => 4096
        self.fc2 = nn.Linear(128*64, 64*64)
        
        # mask is made of 0s/1s so it will just set to 0 any invalid move 4096 => 4096
        self.mask = MaskLayer()

    def forward(self, x, mask=None, debug=False):
        
        # conv1 + bn1 with activation function ReLU
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        
        # conv2 + bn2 with activation function ReLU
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        
        # conv3 + bn3 with activation function ReLU
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        
        # flatten will transform data structure from 3D 8x8x128 to 1D 8192
        x = nn.Flatten()(x)
        
        # fully connected with activation function ReLU
        x = nn.functional.relu(self.fc1(x))
        
        # fully connected WITHOUT ReLU (we want to keep negative values for our output layer)
        x = self.fc2(x)
        
        # if we have a mask we apply it to set to 0 all invalid moves
        if mask is not None:
            x = self.mask(x, mask)
            
        return x