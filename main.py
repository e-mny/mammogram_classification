# from model import DenseNet
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from data_loading.data_loader import createDataLoaders, train_transform
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution
from train.train_loader import train
# from models.create_model import createModel
from models.modelFactory import ModelFactory, printTrainableParams, freezeLayers
from performance.show_graph import plotGraph
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import pickle
from logs.logging import Logger
import time
import argparse

start_time = time.time()

# CLI Parser
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--model', type=str, required=True, help='Model Name')
parser.add_argument('--pretrained', type=bool, default=True, required=False, help='Pretrained Boolean (Default: True)')
parser.add_argument('--dataset', type=str, required=True, help='Datasets: CBIS-DDSM / CMMD / RSNA / USF / VinDr')
parser.add_argument('--num_epochs', type=int, default=200, required=False, help='Number of Epochs (Default: 200)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
MODEL = args.model
DATASET = args.dataset
PRETRAINED_BOOL = args.pretrained
NUM_EPOCHS = args.num_epochs
print(f"MODEL: {MODEL}\t"
    f"DATASET: {DATASET}"
)


## CONSTANTS
SEED = 42
BATCH_SIZE = 32
NUM_CLASSES= 2
LEARNING_RATE = 1e-4
RESAMPLE_RESOLUTION = (224, 224)
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.3    # 30% for validation as of 
# TEST_RATIO = 0.2   # 20% for testing
# VAL_RATIO = 0.1    # 10% for validation
# MODEL = "resnet34"
# DATASET = "CBIS-DDSM" # CBIS-DDSM / CMMD / RSNA / USF / VinDr
# PRETRAINED_BOOL = True
# NUM_EPOCHS = 200

all_images = []
all_labels = []
train_images = []
val_images = []
train_labels = []
val_labels = []
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []
data_folder = os.path.join('/home/emok/sq58_scratch/emok/Data/', DATASET)


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device: {device}")


log_file = Logger(f"./results/{DATASET}_{MODEL}_log.txt")
images_pickle = os.path.join(data_folder, 'image_data.pickle')
labels_pickle = os.path.join(data_folder, 'label_data.pickle')
dataframe_directory = os.path.join(data_folder, "combined_data.csv")
image_folder = os.path.join(data_folder, "images")
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




# # # Save the list to the pickle file
# with open(images_pickle, 'wb') as pickle_file:
#     pickle.dump(np.array(all_images), pickle_file)

# with open(labels_pickle, 'wb') as pickle_file:
#     pickle.dump(np.array(all_labels), pickle_file)

# # Algo 3 (Pickle) (use this for quicker retrieval of data after Algo 1 was done before)

# # Load the list from the pickle file
# with open(images_pickle, 'rb') as image_pickle_file:
#     all_images = pickle.load(image_pickle_file)
# with open(labels_pickle, 'rb') as label_pickle_file:
#     all_labels = pickle.load(label_pickle_file)

# all_images = np.repeat(all_images[:, :, :, np.newaxis], 3, axis = 3)


# print(f"Check if len(all_images) == len(all_labels) == len(df)")
# print(f"all_images: {len(all_images)}")
# print(f"all_labels: {len(all_labels)}") 
# print(f"df: {len(df)}")

train_loader, val_loader = createDataLoaders(batch_size = BATCH_SIZE, train_ratio = TRAIN_RATIO, seed = SEED)

# # Create a model
modelFactoryObj = ModelFactory(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)
model = modelFactoryObj.create_model()

# # Freeze layers
# model = freezeLayers(model)


last_layer = None
for layer in reversed(list(model.children())):
    if isinstance(layer, nn.Linear):
        last_layer = layer
        break
    elif isinstance(layer, nn.Sequential):
        last_layer = layer[-1]
        break

if last_layer is not None:
    num_features = last_layer.out_features
else:
    raise ValueError("No classifier layer found in the pretrained model.")

classifier_layer = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, NUM_CLASSES)
)

class CombinedModel(nn.Module):
    def __init__(self, base_model, classifier):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.classifier = classifier
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

# model = CombinedModel(model, classifier_layer)
# Print model architecture
print(model)
# Print trainable parameters
printTrainableParams(model)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets = train(model, train_loader, val_loader, device, criterion, optimizer, epochs = NUM_EPOCHS)


plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, NUM_EPOCHS, val_preds, val_targets)


historyDict = {
    'train_transform': train_transform,
    'train_accuracy': train_accuracy_history,
    'train_loss': train_loss_history,
    'val_accuracy': val_accuracy_history,
    'val_loss': val_loss_history,
    'val_precision': val_precision_history,
    'val_recall': val_recall_history
}

log_file.log(start_time, historyDict)