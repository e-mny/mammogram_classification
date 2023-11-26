from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.config import *
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from train.train_loader import train
from train.test_loader import test


USE_FEDBN: bool = False

class Client(fl.client.NumPyClient):
    
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

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        _, _, _, _, _, _, _, _, _ = train(self.model, self.trainloader, self.valloader, self.device, self.training_epochs, early_stopping = False)
        # train(self.model, self.trainloader, epochs=self.training_epochs, device=self.device)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        performance_dict = test(self.testloader, self.model, self.device)
        return float(0), self.num_examples["testset"], performance_dict
        
# def train(
#     model: nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     epochs: int,
#     device: torch.device,  # pylint: disable=no-member
# ) -> None:
#     """Train the network."""
#     # Define loss and optimizer
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
#     model.to(device)
#     model.train()

#     print(f"Training {epochs} epoch(s) w/ {len(train_loader)} batches each")


#     for epoch in range(epochs):
#         train_loss = 0.0
#         correct_train = 0
#         total_train = 0
#         for inputs, labels in train_loader:
#             labels = labels.type(torch.FloatTensor)
#             inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             # print(outputs.shape)
#             # print(labels.shape)
#             loss = criterion(outputs.squeeze(), labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             predicted = torch.round(outputs).squeeze()
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()
#         train_accuracy = correct_train / total_train
#         train_loss /= len(train_loader)
        
#         print(f'Epoch [{epoch + 1}/{epochs}]')
#         print(f'Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}')
    
# def test(
#     model: nn.Module,
#     test_loader: torch.utils.data.DataLoader,
#     device: torch.device,  # pylint: disable=no-member
# ) -> Tuple[float, float, float, float, float, float, float]:
#     """Validate the network on the entire test set."""
#     # Define loss and metrics
#     criterion = nn.BCELoss()
#     total = 0
#     correct, loss = 0, 0.0
#     y_truth = []
#     y_predicted = []


#     # Evaluate the network
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for data in test_loader:
#             labels = data[1].type(torch.FloatTensor)
#             images, labels = data[0].to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             outputs = model(images)
#             total += labels.size(0)
#             loss += criterion(outputs.squeeze(), labels).item()
#             # print(outputs)
#             predicted = torch.round(outputs).squeeze()

#             # _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
#             correct += (predicted == labels).sum().item()
#             y_predicted.extend(predicted.cpu().numpy().astype(int))
#             y_truth.extend(labels.cpu().numpy().astype(int))
#     accuracy = correct / total
#     average_loss = loss / len(test_loader)
#     # print(y_predicted)
#     # print(type(y_predicted))
#     # print(y_predicted.shape)
#     # print()
#     # print(y_truth)
#     # print(type(y_truth))
#     # print(y_truth.shape)
#     precision = precision_score(y_truth, y_predicted)
#     recall = recall_score(y_truth, y_predicted)
#     f1_score = 2 * precision * recall / (precision + recall)
#     auc_score = roc_auc_score(y_truth, y_predicted)
#     precision_points, recall_points, _ = precision_recall_curve(y_truth, y_predicted)
#     pr_auc = auc(recall_points, precision_points)
    
#     return average_loss, accuracy, precision, recall, f1_score, auc_score, pr_auc