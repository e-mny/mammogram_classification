from torchvision import transforms
from data_loading.data_augment import createTransforms
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset, CMMDDataset
from data_loading.classDistribution import calcClassDistribution
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from data_loading.displayImage import displaySample
import os
import pydicom
import warnings
# Ignore the specific warning about the number of worker processes
warnings.filterwarnings("ignore", category=UserWarning)



# import multiprocessing
# Get the number of CPU cores
# NUM_CPU_CORES = multiprocessing.cpu_count()

# Set num_workers based on the number of CPU cores
# NUM_WORKERS = min(16, NUM_CPU_CORES)  # You can adjust the maximum value as needed
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0
print(f"NUM_WORKERS: {NUM_WORKERS}")

import numpy as np
import cv2
from PIL import Image


def createDataLoaders(batch_size, dataset, data_augment):
    
    train_transform, val_transform = createTransforms(data_augment)
    print(f"Transforms on Train dataset: {train_transform}")
        
    if dataset == "CBIS-DDSM":
        # Create PyTorch DataLoader
        train_dataset = CBISDataset(form = "mass", mode = "train", transform = train_transform, train = True)
        val_dataset = CBISDataset(form = "mass", mode = "train", transform = val_transform, train = False)
        
        test_dataset = CBISDataset(form = "mass", mode = "test", transform = val_transform)
    elif dataset == "CMMD":
        # Create PyTorch DataLoader
        train_dataset = CMMDDataset(mode = "train", transform = train_transform, train = True)
        val_dataset = CMMDDataset(mode = "train", transform = val_transform, train = False)

        test_dataset = CMMDDataset(mode = "test", transform = val_transform)
    elif dataset == "VinDr":
        # Create PyTorch DataLoader
        train_dataset = VinDrDataset(mode = "train", transform = train_transform, train = True)
        train_dataset = train_dataset.undersample()
        val_dataset = VinDrDataset(mode = "train", transform = val_transform, train = False)
        val_dataset = val_dataset.undersample()

        test_dataset = VinDrDataset(mode = "test", transform = val_transform)
        test_dataset = test_dataset.undersample()
    elif dataset == "RSNA":
        # Create PyTorch DataLoader
        train_dataset = RSNADataset(mode = "train", transform = train_transform, train = True)
        train_dataset = train_dataset.undersample()
        val_dataset = RSNADataset(mode = "train", transform = val_transform, train = False)
        val_dataset = val_dataset.undersample()

        test_dataset = RSNADataset(mode = "test", transform = val_transform)
        test_dataset = test_dataset.undersample()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    print("Created DataLoaders")
    sample_images, sample_titles = displaySample(dataset, train_loader, val_loader, train_transform) # For visualizing transforms
    calcClassDistribution(train_loader, val_loader)
    num_examples = {"trainset": len(train_dataset), "testset": len(val_dataset)}
   
        
    return train_loader, val_loader, test_loader, train_transform, sample_images, sample_titles, num_examples


def createDatasets(dataset, data_augment, val_ratio):
    
    train_transform, val_transform = createTransforms(data_augment)
    print(f"Transforms on Train dataset: {train_transform}")
    transforms = (train_transform, val_transform)
        
        
    print("Creating Datasets")
    if dataset == "CBIS-DDSM":
        # Create PyTorch Datasets
        # combined_dataset = CBISDataset(view = "MLO", mode = "combined", transform = None)
        # combined_dataset = CBISDataset(view = "CC", mode = "combined", transform = None)
        combined_dataset = CBISDataset(view = None, mode = "combined", transform = None)
        X, y = np.array(combined_dataset.data), np.array(combined_dataset.labels)
        return X, y, transforms
    
    elif dataset == "CMMD":
        # Create PyTorch DataLoader
        train_dataset = CMMDDataset(mode = "train", transform = train_transform)
        val_dataset = CMMDDataset(mode = "val", transform = val_transform)
        return train_dataset, val_dataset

   
        

def stratifiedDataLoader(X, y, train_index, val_index, transforms, batch_size):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    train_transforms, val_transforms = transforms

    print(train_transforms)
    print(val_transforms)
    # Create DataLoader objects for training, validation, and test sets
    # train_dataset = CBISCombinedDataset(X_train, y_train, transform=train_transforms)
    # val_dataset = CBISCombinedDataset(X_val, y_val, transform=val_transforms)  # No augmentation for validation
    
    # testNumWorkers(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    
    sample_images, sample_titles = displaySample(train_loader, val_loader, transforms) # For visualizing transforms
    calcClassDistribution(train_loader, val_loader)
    
    
    return train_loader, val_loader, sample_images, sample_titles

def testNumWorkers(train_dataset):
    from time import time
    import multiprocessing as mp

    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(train_dataset ,shuffle=True,num_workers=num_workers, batch_size=1024, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
        
def generateSamplerObject(dataset):
    # Determine the class distribution in the dataset
    labels = dataset.labels
    class_sample_count = defaultdict(int)
    for label in targets:
        class_sample_count[label] += 1

    # Find the minority class size
    minority_class_count = min(class_sample_count.values())

    # Compute weights for each sample
    class_weights = {label: 1.0 / count for label, count in class_sample_count.items()}
    weights = [class_weights[label] for label in targets]

    # Create sampler using WeightedRandomSampler
    return WeightedRandomSampler(weights, len(weights))
    