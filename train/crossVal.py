from sklearn.model_selection import cross_validate
from torch.utils.data import ConcatDataset
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset, CMMDDataset
from data_loading.data_loader import createTransforms
import torch
import numpy as np


def crossValidation(model, dataset, device):
    basic_transform, _ = createTransforms(False)
    scoring_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    if dataset == "CBIS-DDSM":
        # Create PyTorch DataLoader
        train_dataset = CBISDataset(mode = "train", transform = basic_transform)
        val_dataset = CBISDataset(mode = "val", transform = basic_transform)
        
        
    elif dataset == "CMMD":
        # Create PyTorch DataLoader
        train_dataset = CMMDDataset(mode = "train", transform = basic_transform)
        val_dataset = CMMDDataset(mode = "val", transform = basic_transform)


    
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    scores = cross_validate(model, combined_dataset, scoring=scoring_list, n_jobs=-1)
    mean_score = np.mean(scores)
    print(f"Mean Score: {mean_score:.4f}")