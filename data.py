# data.py
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_data(batch_size):
    # Define data loading and transformation
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # for grayscale images
    ])
    
    # Load the dataset using ImageFolder
    train_dataset = ImageFolder("path/to/train", transform=data_transforms)  # for training
    val_dataset = ImageFolder("path/to/validation", transform=data_transforms)  # for validation (tuning hyperparameters)
    test_dataset = ImageFolder("path/to/test", transform=data_transforms)  # for testing
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

