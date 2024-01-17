# models_v1.py
import torch
import torch.nn as nn

# Define the Preprocessing Module
class PreprocessingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PreprocessingModule, self).__init()
        
        # 7x7 Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        
        # 3x3 Max-Pooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

# Define the Basic Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the ResNet-18 Model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init()
        
        # Preprocessing Module
        self.preprocessing = PreprocessingModule(in_channels=3, out_channels=64)
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Residual Blocks
        self.layer1 = self._make_layer(BasicBlock, in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, in_channels=256, out_channels=512, num_blocks=2, stride=2)
        
        # Global Average Pooling Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.preprocessing(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
