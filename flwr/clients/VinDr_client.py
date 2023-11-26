import sys
import os
sys.path.append("/home/emok/sq58/Code/base_mammo")
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from clientObj import Client

# pylint: disable=no-member

# pylint: enable=no-member
from data_loading.data_loader import createDataLoaders
from models.modelFactory import create_model, printTrainableParams
from utils.config import *
from utils.device import initialize_device
import os
DATASET = "VinDr"

class VinDrClient(Client):
    
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
        training_epochs: int,
        device: torch.device
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.training_epochs = training_epochs
        self.device = device


def main() -> None:
    """Load data, start VinDrClient."""
    # Check if GPU is available
    DEVICE = initialize_device()
    

    # Load data
    train_loader, val_loader, test_loader, _, sample_images, sample_titles, num_examples = createDataLoaders(BATCH_SIZE, DATASET, DATA_AUGMENT_BOOL)

    # Load model
    model = create_model(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)
    
    # Print model architecture
    # print(model)
    # Print trainable parameters
    # printTrainableParams(model)
    model = model.to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(train_loader))[0].to(DEVICE))

    # Start client
    client = VinDrClient(model, train_loader, val_loader, test_loader, num_examples, TRAIN_EPOCHS, DEVICE)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
