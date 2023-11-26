from torchvision import transforms


def createTransforms(data_augmentation_bool):
    # Define transformations
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(190),
        transforms.RandomAffine(degrees=0, scale=(1, 1.2)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=90, expand=False),
        # transforms.ColorJitter(brightness = 0.2, contrast= 0.2),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.ToTensor(),
    ])

    val_transform = basic_transform
    
    if data_augmentation_bool:
        return train_transform, val_transform
    else:
        return basic_transform, val_transform