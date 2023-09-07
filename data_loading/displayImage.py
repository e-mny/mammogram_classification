import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import torch
from torchvision import transforms

# def displayRandomSample(batch_size, images, transforms_func):
#     # # Get a random batch from the data loader
#     # random_batch = next(iter(images))
#     # image_array, labels = random_batch
#      # Convert the tensor to a NumPy array and transpose the channels
#     currdatetime = datetime.now()
#     formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")


#     # Batch visualization
#     # Get the batch size and image dimensions
#     # batch = [torch.from_numpy(item).float() for item in random_batch]
#     # batch_size, num_channels, height, width = image_array.shape

#     # Calculate the number of rows and columns for subplots
#     # num_rows = int(np.sqrt(batch_size)) - 1
#     # num_cols = (batch_size + num_rows - 1) // num_rows


#     image_array = images[:batch_size]

#     num_rows = 4
#     num_cols = int(batch_size / num_rows)

#     # Create a figure and a set of subplots
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

#     # Iterate through the batch and plot each image
#     for i in range(batch_size):
#         row = i // num_cols
#         col = i % num_cols
#         # image = image_array[i].permute(1, 2, 0).numpy()  # Transpose channels for display
#         image = image_array[i]  # Transpose channels for display
#         axes[row, col].imshow(image)
#         axes[row, col].axis('off')
        
#     # Adjust layout and display the plots
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f"./data_loading/samples/{formatted_datetime}.png")
    
#     # Iterate through the batch and plot each image
#     for i in range(batch_size):
#         row = i // num_cols
#         col = i % num_cols
#         # image = image_array[i].permute(1, 2, 0).numpy()  # Transpose channels for display
#         image = image_array[i]  # Transpose channels for display
#         image = transforms_func(image)
#         image = image.permute(1, 2, 0).numpy()
#         axes[row, col].imshow(image)
#         axes[row, col].axis('off')

#     # Adjust layout and display the plots
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f"./data_loading/samples/{formatted_datetime}_transformed.png")
#     transformation_list = [transform for transform in transforms_func.transforms if not isinstance(transform, transforms.ToTensor)]

#     # formatted_datetime = formatted_datetime.encode('utf-8')
#     with open("./data_loading/samples/transforms_log.txt", 'a') as f:
#         f.write(formatted_datetime)
#         f.write("\n")
#         f.write(''.join(str(transformation_list)))
#         f.write("\n")
        
        
def getSamples(data_loader):
    negative_class_label = 0
    positive_class_label = 1
    samplesNeg = []
    samplesPos = []
    batch_size = len(next(iter(data_loader)))
    
    for batch in data_loader:
        img_arrs, labels = batch
        for i in range(batch_size):
            img, label = img_arrs[i], labels[i]
            if label.item() == negative_class_label and len(samplesNeg) < 4:
                samplesNeg.append((img, label))
                
            elif label.item() == positive_class_label and len(samplesPos) < 4:
                samplesPos.append((img, label))
                
            print(len(samplesPos), len(samplesNeg))
            if len(samplesNeg) == 4 and len(samplesPos) == 4:
                return samplesNeg, samplesPos
            
    
    return None

# Define a function to display multiple images in a 4x4 grid
def show_images(images, titles, rows, cols, time):

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i, ax in enumerate(axes.ravel()):
        # print(images[i].shape)
        # img = np.transpose(images[i], (1, 2, 0))
        # ax.imshow(images[i], cmap='gray')  # Change the cmap if your images are in color
        ax.imshow(images[i], cmap='gray')  # Change the cmap if your images are in color
        ax.set_title(titles[i])
        ax.axis('off')
    # plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./data_loading/samples/{time}_transformed.png")


def displaySample(train_loader, val_loader, transforms_func):            
    images = []
    titles = []
    formatted_datetime = datetime.now().strftime("%d-%m-%y-%H%M%S")

    
    trainNegSamples, trainPosSamples = getSamples(train_loader)
    valNegSamples, valPosSamples = getSamples(val_loader)
    
    categories = [trainNegSamples, trainPosSamples, valNegSamples, valPosSamples]

    # Loop through the samples to extract images and titles
    for category in categories:
        for sample in category:
            image = sample[0].squeeze().numpy()  # Convert to NumPy array
            image = image.transpose((1, 2, 0))
            target = sample[1].item()  # Extract the target label (0 or 1)
            
            # Define titles based on target labels
            if target == 0:
                titles.append("Benign")
            elif target == 1:
                titles.append("Malignant")

            images.append(image)

    # Show the images in a 4x4 grid
    show_images(images, titles, rows=4, cols=4, time=formatted_datetime)

    transformation_list = [transform for transform in transforms_func.transforms if not isinstance(transform, transforms.ToTensor)]

    with open("./data_loading/samples/transforms_log.txt", 'a') as f:
        f.write(formatted_datetime)
        f.write("\n")
        f.write(''.join(str(transformation_list)))
        f.write("\n")