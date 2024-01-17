# models_v0.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms

class PreprocessingModule(nn.Module):
    # Takes the number of input and output channels for convolutional layer
    def __init__(self, in_channels, out_channels): 
        super(PreprocessingModule, self).__init() #subclass of nn module
        
        # 7x7 Convolutional Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True, adding_mode='zeros', device=None, dtype=None)
        self.relu = nn.ReLU() #supposed to use quadratic linear pooling
        
        # 3x3 Max-Pooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #reduces number of parameters on feature map

    def forward(self, x):
        # Forward pass through the preprocessing module
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x #returns the preprocessed feature map
    


# Put these in a new class CustomResNet18

# Load pretrained ResNet18 with ImageNet
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for 5 classes
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)







# Save trained ResNet
torch.save(model.state_dict(), 'fine_tuned_resnet.pth')