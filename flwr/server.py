"""Flower server example."""

import sys
sys.path.append("/home/emok/sq58/Code/base_mammo")  # Add the parent directory to the Python path
from typing import List, Tuple, OrderedDict, Optional, Dict
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset, CMMDDataset
from data_loading.data_augment import createTransforms
from models.modelFactory import create_model
import torch
from torch.utils.data import DataLoader
from train.test_loader import test
from datetime import datetime
import os
from utils.config import *
from utils.fl import get_model_params
from utils.device import initialize_device

# Check if GPU is available
DEVICE = initialize_device()

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    # Unpack the metrics into separate lists
    num_examples, metric_dicts = zip(*metrics)
    print(f"Metrics: {metrics}")

    # Compute weighted sums for each metric
    weighted_sums = {
        key: sum(num * metric[key] for num, metric in zip(num_examples, metric_dicts)) for key in metric_dicts[0]
    }

    # Calculate weighted averages
    weighted_averages = {key: weighted_sums[key] / sum(num_examples) for key in weighted_sums}

    return weighted_averages

def createValLoader():
    _, val_transform = createTransforms(False)
    cbis_val_dataset = CBISDataset(form = "mass", mode = "train", transform = val_transform, train = False)
    cmmd_val_dataset = CMMDDataset(mode = "train", transform = val_transform, train = False)
    vindr_val_dataset = VinDrDataset(mode = "train", transform = val_transform, train = False)
    rsna_val_dataset = RSNADataset(mode = "train", transform = val_transform, train = False)
        
    vindr_val_dataset = vindr_val_dataset.undersample()
    rsna_val_dataset = rsna_val_dataset.undersample()
    cbis_val_loader = DataLoader(cbis_val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    cmmd_val_loader = DataLoader(cmmd_val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    vindr_val_loader = DataLoader(vindr_val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    rsna_val_loader = DataLoader(rsna_val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    return cbis_val_loader, cmmd_val_loader, vindr_val_loader, rsna_val_loader

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""



    model = create_model(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)
    cbis_val_loader, cmmd_val_loader, vindr_val_loader, rsna_val_loader = createValLoader()

    cbis_name, cmmd_name, vindr_name, rsna_name = cbis_val_loader.dataset.name, cmmd_val_loader.dataset.name, vindr_val_loader.dataset.dataset.name, rsna_val_loader.dataset.dataset.name

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Results in following format: accuracy, loss, precision, recall, f1_score, auc, pr_auc
        cbis_results = [test(cbis_val_loader, model, DEVICE)]
        cmmd_results = [test(cmmd_val_loader, model, DEVICE)]
        vindr_results = [test(vindr_val_loader, model, DEVICE)]
        rsna_results = [test(rsna_val_loader, model, DEVICE)]
        return float(0), {
            cbis_name: cbis_results,
            cmmd_name: cmmd_results,
            vindr_name: vindr_results,
            rsna_name: rsna_results
        }

    return evaluate

def getModelParams():
    model = create_model(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)
    if PRETRAINED_MODEL_PATH:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    return get_model_params(model)

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(),
    initial_parameters = ndarrays_to_parameters(getModelParams())
    )
# strategy = fl.server.strategy.FedAvg()
curr_date = datetime.now().strftime("%d%m%Y-%H%M")
fl.common.logger.configure(identifier="exp1", filename=f"./flwr/logs/{curr_date}_log.txt")


if __name__ == "__main__":
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy
    )
