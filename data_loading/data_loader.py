from torchvision import transforms
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset, CMMDDataset
from collections import Counter
from torch.utils.data import DataLoader
from data_loading.displayImage import displaySample
import os
from data_loading.customTransforms import CLAHETransform, CenterCropWithMainContent
# import multiprocessing

# Get the number of CPU cores
# NUM_CPU_CORES = multiprocessing.cpu_count()

# Set num_workers based on the number of CPU cores
# NUM_WORKERS = min(16, NUM_CPU_CORES)  # You can adjust the maximum value as needed
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 4
# print(f"NUM_WORKERS: {NUM_WORKERS}")

import numpy as np
import cv2
from PIL import Image


def createTransforms(data_augmentation_bool):
    # Define transformations
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # CLAHETransform(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(190),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(degrees=0, scale=(1, 1.2)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45, expand=False),
        # transforms.ColorJitter(brightness = 0.2, contrast= 0.2),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.ToTensor(),
    ])

    val_transform = basic_transform
    
    if data_augmentation_bool:
        return train_transform, val_transform
    else:
        return basic_transform, val_transform

def calcClassDistribution(train_loader, val_loader):
    # Initialize counters for class distribution
    train_class_distribution = Counter()
    val_class_distribution = Counter()

    # Iterate through the training DataLoader to count class occurrences
    for _, labels in train_loader:
        train_class_distribution.update(labels.tolist())

    # Iterate through the validation DataLoader to count class occurrences
    for _, labels in val_loader:
        val_class_distribution.update(labels.tolist())

    # Print class distribution for training DataLoader
    print("Class distribution for training DataLoader:")
    for class_label, count in train_class_distribution.items():
        print(f"Class {class_label}: {count} samples")

    # Print class distribution for validation DataLoader
    print("\nClass distribution for validation DataLoader:")
    for class_label, count in val_class_distribution.items():
        print(f"Class {class_label}: {count} samples")

def createDataLoaders(batch_size, dataset, data_augment):
    
    train_transform, val_transform = createTransforms(data_augment)
    print(f"Transforms on Train dataset: {train_transform}")
        
    if dataset == "CBIS-DDSM":
        # Create PyTorch DataLoader
        train_dataset = CBISDataset(mode = "train", transform = train_transform)
        val_dataset = CBISDataset(mode = "val", transform = val_transform)
    elif dataset == "CMMD":
        # Create PyTorch DataLoader
        train_dataset = CMMDDataset(mode = "train", transform = train_transform)
        val_dataset = CMMDDataset(mode = "val", transform = val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)

    print("Created DataLoaders")
    # print(f"Num of CPU Cores: {os.cpu_count()}")
    sample_images, sample_titles = displaySample(train_loader, val_loader, train_transform) # For visualizing transforms
    calcClassDistribution(train_loader, val_loader)
   
        
    return train_loader, val_loader, train_transform, sample_images, sample_titles
