import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pickle
import os
from datetime import datetime

SAMPLE_INDEX = 0
DATASET = "CMMD" # CBIS-DDSM / CMMD / RSNA / USF / VinDr
data_folder = os.path.join('/home/emok/sq58_scratch/emok/Data/', DATASET)
samples_folder = "home/emok/sq58/Code/base_mammo/data_preprocess/samples"
currdatetime = datetime.now()
formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")
output_file_path = os.path.join(samples_folder, f'{formatted_datetime}.jpeg')

images_pickle = os.path.join(data_folder, 'image_data.pickle')
with open(images_pickle, 'rb') as image_pickle_file:
    all_images = pickle.load(image_pickle_file)
# labels_pickle = os.path.join(data_folder, 'label_data.pickle')
# with open(labels_pickle, 'rb') as label_pickle_file:
#     all_labels = pickle.load(label_pickle_file)

curr_img = all_images[SAMPLE_INDEX]
print(curr_img)
# Define tensor transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert NumPy array to PyTorch tensor
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Apply transformations to the image
image_tensor = transform(curr_img)
print("Transformed")

# Convert tensor back to NumPy array for displaying and saving
curr_img_transformed = image_tensor.numpy().transpose(1, 2, 0)
curr_img_transformed = curr_img_transformed[:, :, 0]
print(curr_img_transformed.shape)

# Display the original and transformed images side by side using matplotlib
plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
# axes[0].plot(x1, y1)
# axes[1].plot(x2, y2)
# Display the original image on the left side
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imread(curr_img)
plt.axis('off')  # Turn off axis labels and ticks

# Display the transformed image on the right side
plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imread(curr_img_transformed)  # Convert tensor layout to match NumPy
plt.axis('off')  # Turn off axis labels and ticks

# plt.tight_layout()  # Ensure proper spacing between subplots
# # plt.show()n

# # Save the transformed image using matplotlib
plt.savefig(output_file_path)

# fig, ax = plt.subplots(1,2)
# ax[0].imread(curr_img)
# ax[0].set_title("Original Image")
# ax[1].imread(curr_img_transformed)
# ax[1].set_title("Transformed Image")
# fig.savefig(output_file_path)

# # Get a random batch from the data loader
#     random_batch = next(iter(data_loader))
#     image_array, labels = random_batch
#     # print(random_batch)


#     # transform = transforms.Compose([
#     #     transforms.ToTensor()
#     # ])
#     # # Convert each PIL image in the list to a tensor
#     # image_tensors = [transform(image) for image in random_batch]

#     # # Convert the list of tensors to a batch tensor
#     # batch = torch.stack(image_tensors)

#     # Select a random sample from the batch
#     random_sample = image_array[sample_index]

#     # Convert the tensor to a NumPy array and transpose the channels
#     image = random_sample.permute(1, 2, 0).numpy()
#     image = image[:, :, 0]

#     print(image.shape)
#     # Display the image using matplotlib
#     plt.imshow(image)
#     plt.axis('off')  # Turn off axis labels and ticks
#     currdatetime = datetime.now()
#     formatted_datetime = currdatetime.strftime("%d-%m-%y-%H%M%S")
#     # plt.savefig(f"./samples/{formatted_datetime}.png")
#     # image.save(os.path.join("home/emok/sq58/Code/base_mammo/data_loading/samples", f"{formatted_datetime}.png"))
#     cv2.imwrite(f"./data_loading/samples/{formatted_datetime}.png", image)


#     # Batch visualization
#     # Get the batch size and image dimensions
#     # batch = [torch.from_numpy(item).float() for item in random_batch]
#     batch_size, num_channels, height, width = image_array.shape

#     # Calculate the number of rows and columns for subplots
#     num_rows = int(np.sqrt(batch_size))
#     num_cols = (batch_size + num_rows - 1) // num_rows

#     # Create a figure and a set of subplots
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

#     # Iterate through the batch and plot each image
#     for i in range(batch_size):
#         row = i // num_cols
#         col = i % num_cols
#         image = image_array[i].permute(1, 2, 0).numpy()  # Transpose channels for display
#         axes[row, col].imshow(image)
#         axes[row, col].axis('off')

#     # Adjust layout and display the plots
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(f"./data_loading/samples/{formatted_datetime}.png")
