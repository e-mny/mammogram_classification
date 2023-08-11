import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms

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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def createDataLoaders(all_images, all_labels, training_ratio, val_ratio, batch_size):
    # Append normalized values to train_images
    train_upper_lim = int(training_ratio * len(all_images))
    train_images = np.array(all_images[:train_upper_lim], dtype=np.float32)
    train_labels = np.array(all_labels[:train_upper_lim], dtype=np.int8)

    # Append normalized values to val_images
    val_upper_lim = int((training_ratio + val_ratio) * len(all_images))
    val_images = np.array(all_images[train_upper_lim:val_upper_lim], dtype=np.float32)
    val_labels = np.array(all_labels[train_upper_lim:val_upper_lim], dtype=np.int8)


    # print(f"train_images: {train_images}\n\n")
    # print(f"train_labels: {train_labels}\n\n")

    # Convert lists to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.int8)
    val_images = torch.tensor(val_images, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.int8)

    # Create PyTorch DataLoader
    train_dataset = CustomDatasetClass(train_images, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CustomDatasetClass(val_images, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Created DataLoaders")

    return train_loader, val_loader


