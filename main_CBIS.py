# from model import DenseNet
import pickle
import torch
import pandas as pd
import numpy as np
import os
from data_loading.data_loader import createDataLoaders
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution
from train.train_loader import train
from models.create_model import createModel
from performance.show_graph import plotGraph
import pickle


## CONSTANTS
BATCH_SIZE = 32
NUM_CLASSES= 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 300
all_images = []
all_labels = []
train_images = []
val_images = []
train_labels =[]
val_labels =[]
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []
data_folder = '/home/emok/sq58_scratch/emok/Data/CBIS-DDSM/'
dataset_name = data_folder.split("/")[-2]
RESAMPLE_RESOLUTION = (224, 224)
TRAIN_RATIO = 0.7  # 70% for training
TEST_RATIO = 0.2   # 20% for testing
VAL_RATIO = 0.1    # 10% for validation

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")



images_pickle = os.path.join(data_folder, 'image_data.pickle')
labels_pickle = os.path.join(data_folder, 'label_data.pickle')
dataframe_directory = os.path.join(data_folder, "combined_data.csv")
df = pd.read_csv(dataframe_directory)
# print(df)

# Algo 1 (run this if first time)
# for i in range(len(df)):
#     print(i)
#     # print(f"Index: {i}, folder_name: {df['folder_name'].iloc[i]}, pathology: {df['pathology'].iloc[i]}, class_label: {df['class_label'].iloc[i]}")
#     curr_folder = os.path.join(data_folder, df['folder_name'].iloc[i])
#     curr_label = df['class_label'].iloc[i]
    
#     for root, _, files in os.walk(curr_folder):
#         # print(root)
#         for file in files:
#             if file.lower().endswith('.dcm'):
#                 curr_file_path = os.path.join(root, file)
#                 pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
#                 normalized_data = normalize_intensity(pixel_data)
#                 resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
#                 all_images.append(resampled_image)
#                 all_labels.append(curr_label)




# # Save the list to the pickle file
# with open(images_pickle, 'wb') as pickle_file:
#     pickle.dump(np.array(all_images), pickle_file)

# with open(labels_pickle, 'wb') as pickle_file:
#     pickle.dump(np.array(all_labels), pickle_file)

# # Algo 3 (Pickle) (use this for quicker retrieval of data after Algo 1 was done before)

# # Load the list from the pickle file
with open(images_pickle, 'rb') as image_pickle_file:
    all_images = pickle.load(image_pickle_file)
with open(labels_pickle, 'rb') as label_pickle_file:
    all_labels = pickle.load(label_pickle_file)




print(f"Check if len(all_images) == len(all_labels) == len(df) {len(all_images) == len(all_labels) == len(df)}")
print("Appended all_labels, and all_images")

# print(f"all_images: {all_images}\n\n")
# print(f"all_labels: {all_labels}\n\n")

train_loader, val_loader = createDataLoaders(all_images, all_labels, training_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, batch_size = BATCH_SIZE)

model, criterion, optimizer = createModel(num_input_channels = 1, num_classes=NUM_CLASSES, lr=LEARNING_RATE, pretrained=False)

train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_preds, val_targets = train(model, train_loader, val_loader, device, criterion, optimizer, epochs = NUM_EPOCHS)

print(f"---Averages---")
print(f"Train Accuracy: {sum(train_accuracy_history) / len(train_accuracy_history)}")
print(f"Train Loss: {sum(train_loss_history) / len(train_loss_history)}")
print(f"Validation Accuracy: {sum(val_accuracy_history) / len(val_accuracy_history)}")
print(f"Validation Loss: {sum(val_loss_history) / len(val_loss_history)}")

plotGraph(dataset_name, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, NUM_EPOCHS, val_preds, val_targets)
