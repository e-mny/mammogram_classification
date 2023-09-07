from torchvision import transforms
from data_loading.datasets import CBISDataset, RSNADataset, VinDrDataset
from collections import Counter
from torch.utils.data import DataLoader
from data_loading.displayImage import displaySample


# Define transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # transforms.ColorJitter(brightness = 0.2, contrast= 0.2),
    # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=45, expand=False),
    transforms.Resize((224, 224)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


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

def createDataLoaders(batch_size, train_ratio, seed):

    print(f"Transforms on Train dataset: {train_transform}")
    # Create PyTorch DataLoader
    dataset = CBISDataset(transform=train_transform)
    train_dataset, val_dataset = dataset.split(split_ratio = train_ratio, random_seed = seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    print("Created DataLoaders")
    displaySample(train_loader, val_loader, train_transform) # For visualizing transforms
    calcClassDistribution(train_loader, val_loader)

    
        
    return train_loader, val_loader