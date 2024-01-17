# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data, preprocess_data
from models_v1 import CustomResNet18, PreprocessingModule
from train import train_model, evaluate_model

# Define hyperparameters and configurations
learning_rate = 1e-4 #rate at which loss function is modified
batch_size = 2 #number of data samples processed in each forward and backward pass. This computer probably can't handle a lot.
num_epochs = 30 #30 runs through data set

# Load data
train_loader, val_loader, test_loader = load_data(batch_size)

# Preprocess data
preprocessing_module = PreprocessingModule(in_channels=3, out_channels=64)

# Initialize the model
device = torch.device("cpu") #don't have a gpu
model = CustomResNet18(num_classes=3).to(device)

# Define loss function and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, device)

# Evaluate the model on the test set
test_accuracy, test_loss = evaluate_model(model, test_loader, loss, device)
