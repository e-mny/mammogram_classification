import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import torch
from torchvision import transforms

def displayRandomSample(batch_size, images, transforms_func):
    # # Get a random batch from the data loader
    # random_batch = next(iter(images))
    # image_array, labels = random_batch
     # Convert the tensor to a NumPy array and transpose the channels
    currdatetime = datetime.now()
    formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")


    # Batch visualization
    # Get the batch size and image dimensions
    # batch = [torch.from_numpy(item).float() for item in random_batch]
    # batch_size, num_channels, height, width = image_array.shape

    # Calculate the number of rows and columns for subplots
    # num_rows = int(np.sqrt(batch_size)) - 1
    # num_cols = (batch_size + num_rows - 1) // num_rows


    image_array = images[:batch_size]

    num_rows = 4
    num_cols = int(batch_size / num_rows)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Iterate through the batch and plot each image
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # image = image_array[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        image = image_array[i]  # Transpose channels for display
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        
    # Adjust layout and display the plots
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./data_loading/samples/{formatted_datetime}.png")
    
    # Iterate through the batch and plot each image
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        # image = image_array[i].permute(1, 2, 0).numpy()  # Transpose channels for display
        image = image_array[i]  # Transpose channels for display
        image = transforms_func(image)
        image = image.permute(1, 2, 0).numpy()
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    # Adjust layout and display the plots
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./data_loading/samples/{formatted_datetime}_transformed.png")
    transformation_list = [transform for transform in transforms_func.transforms if not isinstance(transform, transforms.ToTensor)]

    # formatted_datetime = formatted_datetime.encode('utf-8')
    with open("./data_loading/samples/transforms_log.txt", 'a') as f:
        f.write(formatted_datetime)
        f.write("\n")
        f.write(''.join(str(transformation_list)))
        f.write("\n")