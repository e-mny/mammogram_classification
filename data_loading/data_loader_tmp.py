import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from data_loading.displayImage import displayRandomSample
import torchvision.transforms.functional as F

class CustomDatasetClass(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # print(len(self.images))
        return len(self.images)

    def __getitem__(self, idx):
        # print(idx)
        image = self.images[idx].numpy()
        # print(image)
        label = self.labels[idx]

        if self.transform:
            transformed_sample = self.transform(image)
            return transformed_sample, label
        else:
            return image, label

# Define transformations
train_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness = 2, contrast= 2.0),
    # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=45, expand=True),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224))
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

def createDataLoaders(all_images, all_labels, training_ratio, val_ratio, batch_size):
    # Append normalized values to train_images
    train_upper_lim = int(training_ratio * len(all_images))
    train_images = np.array(all_images[:train_upper_lim], dtype=np.float32)
    train_labels = np.array(all_labels[:train_upper_lim], dtype=np.int8)

    # TRAIN / VALIDATION / TEST
    # # Append normalized values to val_images
    # val_upper_lim = int((training_ratio + val_ratio) * len(all_images))
    # val_images = np.array(all_images[train_upper_lim:val_upper_lim], dtype=np.float32)
    # val_labels = np.array(all_labels[train_upper_lim:val_upper_lim], dtype=np.int8)


    # TRAIN / VALIDATION
    val_images = np.array(all_images[train_upper_lim:], dtype=np.float32)
    val_labels = np.array(all_labels[train_upper_lim:], dtype=np.int8)



    # print(f"train_images: {train_images}\n\n")
    # print(f"train_labels: {train_labels}\n\n")

    # Convert lists to PyTorch tensors
    train_images = torch.Tensor(train_images).float()
    train_labels = torch.Tensor(train_labels).long()
    val_images = torch.Tensor(val_images).float()
    val_labels = torch.Tensor(val_labels).long()

    # Calculate class frequencies
    train_class_counts = np.bincount(train_labels)
    val_class_counts = np.bincount(val_labels)

    # Print class distribution
    print("Training class distribution:")
    for class_idx, count in enumerate(train_class_counts):
        curr_class = "Malignant" if class_idx == 1 else "Benign"
        print(f"Class {class_idx} ({curr_class}): {count} samples")

    print("\nValidation class distribution:")
    for class_idx, count in enumerate(val_class_counts):
        curr_class = "Malignant" if class_idx == 1 else "Benign"
        print(f"Class {class_idx} ({curr_class}): {count} samples")


    print(f"Transforms on Train dataset: {train_transform}")
    # Create PyTorch DataLoader
    train_dataset = CustomDatasetClass(train_images, train_labels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetClass(val_images, val_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Created DataLoaders")
    displaySample(batch_size, all_images, train_transform) # For visualizing transforms

    return train_loader, val_loader
