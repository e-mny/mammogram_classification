import torchvision
import torch.nn as nn
import torch.optim as optim

def createModel(num_input_channels, num_classes, lr, pretrained):
    model = torchvision.models.resnet18(pretrained=pretrained)

    # Replace the first convolutional layer to accept single-channel input
    model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Created model, criterion, optimizer")
    return model, criterion, optimizer
