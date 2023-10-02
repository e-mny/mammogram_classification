import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import os
from data_loading.data_loader import createDataLoaders, createDatasets, calcClassDistribution, stratifiedDataLoader, CombinedDataset
from train.train_loader import train, stratified_train, reset_weights
# from models.create_model import createModel
from models.modelFactory import ModelFactory, printTrainableParams, freezeLayers
from performance.show_graph import plotGraph
from sklearn.model_selection import StratifiedShuffleSplit
from visualization.explainPred import generateHeatMap
import pickle
from logs.logging import Logger
import time
import argparse
import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()
start_time = time.time()

# CLI Parser
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Description of your script.")

# Add arguments
parser.add_argument('--model', type=str, required=True, help='Model Name')
parser.add_argument('--pretrained', type=bool, default=True, required=False, help='Pretrained Boolean (Default: True)')
parser.add_argument('--dataset', type=str, required=True, help='Datasets: CBIS-DDSM / CMMD / RSNA / USF / VinDr')
parser.add_argument('--num_epochs', type=int, default=200, required=False, help='Number of Epochs (Default: 200)')
parser.add_argument('--data_augment', action='store_true', help='Refer to data_loading/data_loader.py for list of augmentation')
parser.add_argument('--no-data_augment', dest='data_augment', action='store_false')
parser.set_defaults(data_augment=False)

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
MODEL = args.model
DATASET = args.dataset
PRETRAINED_BOOL = args.pretrained
NUM_EPOCHS = args.num_epochs
DATA_AUGMENT_BOOL = args.data_augment
print(f"MODEL: {MODEL}\t"
    f"DATASET: {DATASET}\t"
    f"DATA AUGMENT: {DATA_AUGMENT_BOOL}"
)


## CONSTANTS
SEED = 42
BATCH_SIZE = 32
NUM_CLASSES= 2
LEARNING_RATE = 1e-4
RESAMPLE_RESOLUTION = (224, 224)
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.3    # 30% for validation
BASE_MODEL_PATH = "./models/base_model.pth"
# TEST_RATIO = 0.2   # 20% for testing
# VAL_RATIO = 0.1    # 10% for validation

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
print(f"Device: {device}")
log_file = Logger(f"./results/{DATASET}_{MODEL}_log.txt")



# print(df)
# images_pickle = os.path.join(data_folder, 'image_data.pickle')
# labels_pickle = os.path.join(data_folder, 'label_data.pickle')
# dataframe_directory = os.path.join(data_folder, "combined_data.csv")
# image_folder = os.path.join(data_folder, "images")
# df = pd.read_csv(dataframe_directory)

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
# if __name__ == "__main__":

# # Create a model
modelFactoryObj = ModelFactory(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)
model = modelFactoryObj.create_model()


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

model = CombinedModel(model, classifier_layer)
# Print model architecture
print(model)
# Print trainable parameters
printTrainableParams(model)
torch.save(model.state_dict(), BASE_MODEL_PATH)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Non-Kfold Cross Validation
# train_loader, val_loader, transforms, sample_images, sample_titles = createDataLoaders(batch_size = BATCH_SIZE, dataset = DATASET, data_augment = DATA_AUGMENT_BOOL)
# train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets = train(model, train_loader, val_loader, device, criterion, optimizer, epochs = NUM_EPOCHS)
# historyDict = {
#     'train_transform': transforms,
#     'train_accuracy': train_accuracy_history,
#     'train_loss': train_loss_history,
#     'val_accuracy': val_accuracy_history,
#     'val_loss': val_loss_history,
#     'val_precision': val_precision_history,
#     'val_recall': val_recall_history
# }

# K-fold Cross Validation
train_dataset, val_dataset, transforms = createDatasets(dataset = DATASET, data_augment = DATA_AUGMENT_BOOL)

combined_dataset = CombinedDataset(train_dataset, val_dataset, transforms)

split_train_accuracy_history = []
split_train_loss_history = []
split_val_accuracy_history = []
split_val_loss_history = []
split_val_precision_history = []
split_val_recall_history = []

sss = StratifiedShuffleSplit(n_splits = 5, train_size = TRAIN_RATIO, random_state = SEED)
for i, (train_index, val_index) in enumerate(sss.split(combined_dataset.data, combined_dataset.labels)):
    model.load_state_dict(torch.load(BASE_MODEL_PATH))
    print(f"--------FOLD {i}--------")
    start_fold_time = time.time()
    train_loader, val_loader, sample_images, sample_titles = stratifiedDataLoader(combined_dataset, train_index, val_index, transforms, BATCH_SIZE)
    train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets, early_stopped_epoch = stratified_train(model, train_loader, val_loader, device, criterion, optimizer, epochs = NUM_EPOCHS)
    
    train_acc = train_accuracy_history[-1]
    train_loss = train_loss_history[-1]
    val_acc = val_accuracy_history[-1]
    val_loss = val_loss_history[-1]
    val_precision = val_precision_history[-1]
    val_recall = val_recall_history[-1]
    
    split_train_accuracy_history.append(train_acc)
    split_train_loss_history.append(train_loss)
    split_val_accuracy_history.append(val_acc)
    split_val_loss_history.append(val_loss)
    split_val_precision_history.append(val_precision)
    split_val_recall_history.append(val_recall)
    generateHeatMap(sample_images, sample_titles, model, device)
    
    print(f"Time taken for fold {i}: {time.time() - start_fold_time}")

    historyDict = {
        'folds': i,
        'train_transform': transforms,
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'val_precision': val_precision,
        'val_recall': val_recall
    }

    log_file.log(start_fold_time, historyDict)
    if early_stopped_epoch < NUM_EPOCHS:
        plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, early_stopped_epoch, val_preds, val_targets)
    else:
        plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, NUM_EPOCHS, val_preds, val_targets)

average_train_acc = np.mean(split_train_accuracy_history)
average_train_loss = np.mean(split_train_loss_history)
average_val_acc = np.mean(split_val_accuracy_history)
average_val_loss = np.mean(split_val_loss_history)
average_precision = np.mean(split_val_precision_history)
average_recall = np.mean(split_val_recall_history)

finalHistoryDict = {
    'folds': None,
    'train_transform': transforms,
    'train_accuracy': average_train_acc,
    'train_loss': average_train_loss,
    'val_accuracy': average_val_acc,
    'val_loss': average_val_loss,
    'val_precision': average_precision,
    'val_recall': average_recall
}

log_file.log(start_time, finalHistoryDict)
print(f"\n{finalHistoryDict}")

pr.disable()

s = io.StringIO()
results = pstats.Stats(pr, stream=s)
results.sort_stats(pstats.SortKey.TIME)
results.print_stats()
with open("./profile.txt", "w") as f:
    f.write(s.getvalue())
    