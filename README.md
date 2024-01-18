CNN prototype for analyzing the quality of ultrasound images. Uses modified ResNet18 architecture.

Main:
Handles training and testing the CNN. The model is trained on the cpu (lacking cuda GPU). Mean Squared Error (MSE) is used as loss function, and Adam optimizer with specified learning rate. The trained model is evaluated on the test set, accuracy and loss are printed.

Data:
Loading function load the dataset and create data loaders for training, validation, and testing. Uses 'compose' to normalize dataset images. Test data is not randomly cropped.

Model:
Contains the ResNet 18 architecture with the preprocessing module, which has a 7x7 convolutional layer, ReLU activation, and 3x3 max-pooling layer. The basic block is the basic residual block used in ResNet architecture, with two convolutional layers with batch normalization and a residual connection.