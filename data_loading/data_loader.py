from torchvision import transforms
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset, CMMDDataset, CBISNewDataset, CBISCombinedDataset
from collections import Counter
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset
from data_loading.displayImage import displaySample
import os
import pydicom
# import multiprocessing
import warnings

# Ignore the specific warning about the number of worker processes
warnings.filterwarnings("ignore", category=UserWarning)


# Get the number of CPU cores
# NUM_CPU_CORES = multiprocessing.cpu_count()

# Set num_workers based on the number of CPU cores
# NUM_WORKERS = min(16, NUM_CPU_CORES)  # You can adjust the maximum value as needed
# NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 6
print(f"NUM_WORKERS: {NUM_WORKERS}")

import numpy as np
import cv2
from PIL import Image

# class CombinedDataset(Dataset):
#     def __init__(self, dataset_a, dataset_b, transform):
#         self.dataset_a = dataset_a
#         self.dataset_b = dataset_b

#         # Calculate the total length of the combined dataset
#         self.total_length = len(self.dataset_a) + len(self.dataset_b)
#         self.data = self.dataset_a.data + self.dataset_b.data
#         self.labels = self.dataset_a.labels + self.dataset_b.labels
#         self.transform = transform

#     def __len__(self):
#         return self.total_length

#     def __getitem__(self, idx):
#         # if idx >= len(self.dataset_a):
#         #     # Subtract the length of the first dataset to get the index in the second dataset
#         #     idx -= len(self.dataset_a)
#         #     image_path = self.dataset_b.data[idx]
#         #     label = self.dataset_b.labels[idx]
#         # else:
#         #     image_path = self.dataset_a.data[idx]
#         #     label = self.dataset_a.labels[idx]
#         image_path = self.data[idx]
#         label = self.labels[idx]
        
            
        
#         # JPEG
#         imageArr = self.load_jpeg(image_path)
#         # imageArr = self.stack_jpeg(imageArr)

#         if self.transform:
#             imageArr = self.transform(imageArr)

#         return imageArr, label
    
#     def load_dicom(self, dicom_path):
#         ds = pydicom.dcmread(dicom_path)
#         image = ds.pixel_array

#         # Convert to 8-bit grayscale
#         image = image - np.min(image)
#         image = (image / np.max(image) * 255).astype(np.uint8)

#         imagearray = Image.fromarray(image.astype(float))
#         return imagearray
    
#     def load_jpeg(self, jpeg_path):
#         image = Image.open(jpeg_path).convert("RGB")
        
#         # image = np.array(image)
        
#         return image
    
#     def stack_jpeg(self, image):
#         stacked_image = np.stack((image,) * 3, axis=-1)
#         return stacked_image


def createTransforms(data_augmentation_bool):
    # Define transformations
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(190),
        # transforms.RandomAffine(degrees=0, scale=(1, 1.2)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90, expand=False),
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

def createDatasets(dataset, data_augment):
    
    train_transform, val_transform = createTransforms(data_augment)
    print(f"Transforms on Train dataset: {train_transform}")
    transforms = (train_transform, val_transform)
        
        
    if dataset == "CBIS-DDSM":
        # Create PyTorch Datasets
        # combined_dataset = CBISDataset(view = "MLO", mode = "combined", transform = None)
        # combined_dataset = CBISDataset(view = "CC", mode = "combined", transform = None)
        combined_dataset = CBISDataset(view = None, mode = "combined", transform = None)
        X, y = np.array(combined_dataset.data), np.array(combined_dataset.labels)
    elif dataset == "CBIS-DDSM_new":
        # Create PyTorch DataLoader
        train_dataset = CBISNewDataset(mode = "train", transform = train_transform)
        val_dataset = CBISNewDataset(mode = "val", transform = val_transform)
    elif dataset == "CMMD":
        # Create PyTorch DataLoader
        train_dataset = CMMDDataset(mode = "train", transform = train_transform)
        val_dataset = CMMDDataset(mode = "val", transform = val_transform)

    print("Created Datasets")
   
        
    return X, y, transforms

def stratifiedDataLoader(X, y, train_index, val_index, transforms, batch_size):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    train_transforms, val_transforms = transforms

    print(train_transforms)
    print(val_transforms)
    # Create DataLoader objects for training, validation, and test sets
    train_dataset = CBISCombinedDataset(X_train, y_train, transform=train_transforms)
    val_dataset = CBISCombinedDataset(X_val, y_val, transform=val_transforms)  # No augmentation for validation
    
    # testNumWorkers(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    
    sample_images, sample_titles = displaySample(train_loader, val_loader, transforms) # For visualizing transforms
    calcClassDistribution(train_loader, val_loader)
    
    
    return train_loader, val_loader, sample_images, sample_titles