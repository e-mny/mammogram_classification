"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223


from typing import Tuple, Dict
import sys
sys.path.append("/home/emok/sq58/Code/base_mammo")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision import models
from data_loading.data_loader import createDataLoaders
from models.modelFactory import create_model, printTrainableParams
from clientObj import train, test
from utils.config import *
from utils.device import initialize_device
import os



# pylint: disable=unsubscriptable-object
# class Net(nn.Module):
#     """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.bn3 = nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120, 84)
#         self.bn4 = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 10)

#     # pylint: disable=arguments-differ,invalid-name
#     def forward(self, x: Tensor) -> Tensor:
#         """Compute forward pass."""
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.bn3(self.fc1(x)))
#         x = F.relu(self.bn4(self.fc2(x)))
#         x = self.fc3(x)
#         return x

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.base_model = self.freezeModel(models.resnet50(pretrained=True))
        
        self.num_features = self.base_model.fc.out_features
        self.classifier_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def freezeModel(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier_layer(x)
        return x

# def load_data() -> (
#     Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]
# ):
#     """Load CIFAR-10 (training and test set)."""
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
#     testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
#     num_examples = {"trainset": len(trainset), "testset": len(testset)}
#     return trainloader, testloader, num_examples

# def load_custom_data() -> (
#     Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]
# ):
#     """Load CBIS-DDSM dataset."""
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees=45),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#     trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
#     testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
#     num_examples = {"trainset": len(trainset), "testset": len(testset)}
#     return trainloader, testloader, num_examples


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Data Augmentation: {DATA_AUGMENT_BOOL}")
    print("---------------Centralized PyTorch training--------------")
    print("Load data")
    train_loader, test_loader, _, train_transform, sample_images, sample_titles, num_examples = createDataLoaders(batch_size = BATCH_SIZE, dataset = "CBIS-DDSM", data_augment = DATA_AUGMENT_BOOL)
    net = CustomResNet50(num_classes=2).to(DEVICE)
    print("Start training")
    train(model=net, train_loader=train_loader, epochs=TRAIN_EPOCHS, device=DEVICE)
    print("Evaluate model")
    loss, accuracy, precision, recall, f1_score, auc_score, pr_auc = test(model=net, test_loader=test_loader, device=DEVICE)
    # loss, accuracy = test(model=net, testloader=test_loader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)
    print("AUC Score: ", auc_score)
    print("PRAUC Score: ", pr_auc)


if __name__ == "__main__":
    main()
