
from collections import Counter

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