import torchvision
import torch.nn as nn
import torch.optim as optim

def createModel(num_input_channels, num_classes, lr, pretrained):
    # Define your DenseNet model
    model = torchvision.models.densenet121(pretrained=pretrained) # NO PRETRAINED MODELS
    model.features.conv0 = nn.Conv2d(in_channels=num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) # To adjust input size

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    print("Created model, criterion, optimizer")
    return model, criterion, optimizer
