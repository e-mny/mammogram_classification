import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import torch
from torchvision import transforms


def getSamples(data_loader):
    negative_class_label = 0
    positive_class_label = 1
    samplesNeg = []
    samplesPos = []
    batch_size = data_loader.batch_size
    
    for batch in data_loader:
        img_arrs, labels = batch
        for i in range(batch_size):
            img, label = img_arrs[i], labels[i]
            if label.item() == negative_class_label and len(samplesNeg) < 4:
                samplesNeg.append((img, label))
                
            elif label.item() == positive_class_label and len(samplesPos) < 4:
                samplesPos.append((img, label))
                
            # print(len(samplesPos), len(samplesNeg))
            if len(samplesNeg) == 4 and len(samplesPos) == 4:
                return samplesNeg, samplesPos
            
    
    return None

# Define a function to display multiple images in a 4x4 grid
def show_images(images, titles, rows, cols, time):

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i, ax in enumerate(axes.ravel()):
        # print(images[i].shape)
        # img = np.transpose(images[i], (1, 2, 0))
        ax.imshow(images[i], cmap='gray')  # Change the cmap if your images are in color
        ax.set_title(titles[i])
        ax.axis('off')
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./data_loading/samples/{time}_transformed.png")


def displaySample(train_loader, val_loader, transforms_func):            
    sample_images = []
    sample_titles = []
    formatted_datetime = datetime.now().strftime("%d-%m-%y-%H%M%S")
    train_transforms, _ = transforms_func
    print("Creating Samples")
    
    trainNegSamples, trainPosSamples = getSamples(train_loader)
    valNegSamples, valPosSamples = getSamples(val_loader)
    
    categories = [trainNegSamples, trainPosSamples, valNegSamples, valPosSamples]

    # Loop through the samples to extract images and titles
    for category in categories:
        for sample in category:
            image = sample[0].squeeze().numpy()  # Convert to NumPy array
            # image = (image / np.max(image) * 255).astype(np.uint8)
            image = image.transpose((1, 2, 0))
            # print(image.shape)
            target = sample[1].item()  # Extract the target label (0 or 1)
            
            # Define titles based on target labels
            if target == 0:
                sample_titles.append("Benign")
            elif target == 1:
                sample_titles.append("Malignant")

            sample_images.append(image)

    # Show the images in a 4x4 grid
    show_images(sample_images, sample_titles, rows=4, cols=4, time=formatted_datetime)

    transformation_list = [transform for transform in train_transforms.transforms if not isinstance(transform, transforms.ToTensor)]

    with open("./data_loading/samples/transforms_log.txt", 'a') as f:
        f.write(formatted_datetime)
        f.write("\n")
        f.write(''.join(str(transformation_list)))
        f.write("\n")
        
    print("Created Samples")
    return sample_images, sample_titles