import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from data_loading.data_loader import createDataLoaders, createDatasets, calcClassDistribution, stratifiedDataLoader
from train.train_loader import train, stratified_train
# from models.create_model import createModel
from models.modelFactory import ModelFactory, printTrainableParams, freezeLayers, reset_weights
from performance.show_graph import plotGraph
from sklearn.model_selection import StratifiedShuffleSplit
from visualization.explainPred import generateHeatMap
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
parser.add_argument('--early_stopping', action='store_true', required=False, help='Stop training if validation loss doesnt decrease')
parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
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
EARLY_STOPPING_BOOL = args.early_stopping
DATA_AUGMENT_BOOL = args.data_augment
print(f"MODEL: {MODEL}\t"
    f"DATASET: {DATASET}\t"
    f"DATA AUGMENT: {DATA_AUGMENT_BOOL}"
)


## CONSTANTS
NUM_FOLDS = 1
BATCH_SIZE = 512
SEED = 42
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
RESAMPLE_RESOLUTION = (224, 224)
BASE_MODEL_PATH = "./models/base_model.pth"

# Use default train/test split
# Train/validation Ratio: 80/20
TRAIN_RATIO = 0.8  # 80% for training
VAL_RATIO = 0.2    # 20% for validation

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
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {DEVICE}")
log_file = Logger(f"./results/{DATASET}_{MODEL}_log.txt")

# if __name__ == "__main__":

# Create a model object
modelFactoryObj = ModelFactory(model_name=MODEL, num_classes=NUM_CLASSES, input_channels=3, pretrained=PRETRAINED_BOOL)


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
# CBIS-Dataset
# X, y, transforms = createDatasets(dataset = DATASET, data_augment = DATA_AUGMENT_BOOL)
# CBIS-New-Dataset (Testing purposes)
# train_data, val_data = createDatasets(dataset = DATASET, data_augment = DATA_AUGMENT_BOOL, val_ratio = VAL_RATIO)

split_train_accuracy_history = []
split_train_loss_history = []
split_val_accuracy_history = []
split_val_loss_history = []
split_val_precision_history = []
split_val_recall_history = []
split_val_f1_history = []

LR_LIST = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
BATCHSIZE_LIST = [2, 4, 8, 16]
best_metric = np.inf
best_lr = 0
best_batchsize = 0

sss = StratifiedShuffleSplit(n_splits = NUM_FOLDS, train_size = TRAIN_RATIO, random_state = SEED)
# for i, (train_index, val_index) in enumerate(sss.split(X, y), start=1):
# for i, (train_index, val_index) in enumerate(sss.split(train_data, val_data), start=1):
model = modelFactoryObj.create_model()
model = model.to(DEVICE)

# Print model architecture
print(model)
# Print trainable parameters
printTrainableParams(model)

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss() # If using linear.out_features = 2
criterion = nn.BCELoss() # If using sigmoid
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)


# for LEARNING_RATE in LR_LIST:
# for BATCH_SIZE in BATCHSIZE_LIST:
# model.load_state_dict(torch.load(BASE_MODEL_PATH))
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# reset_weights(model)
# print(f"--------FOLD {i}--------")
start_fold_time = time.time()

# Testing
train_loader, val_loader, train_transform, sample_images, sample_titles = createDataLoaders(BATCH_SIZE, DATASET, DATA_AUGMENT_BOOL, VAL_RATIO)
transforms = train_transform
train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets, early_stopped_epoch = stratified_train(model, train_loader, val_loader, DEVICE, criterion, optimizer, epochs = NUM_EPOCHS, early_stopping = EARLY_STOPPING_BOOL)


# NEW
# train_loader, val_loader, sample_images, sample_titles = stratifiedDataLoader(X, y, train_index, val_index, transforms, BATCH_SIZE)
# train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets, early_stopped_epoch = stratified_train(model, train_loader, val_loader, DEVICE, criterion, optimizer, epochs = NUM_EPOCHS, early_stopping = EARLY_STOPPING_BOOL)

train_acc = train_accuracy_history[-1]
train_loss = train_loss_history[-1]
val_acc = val_accuracy_history[-1]
val_loss = val_loss_history[-1]
val_precision = val_precision_history[-1]
val_recall = val_recall_history[-1]
# Calculating F1 score
val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)

split_train_accuracy_history.append(train_acc)
split_train_loss_history.append(train_loss)
split_val_accuracy_history.append(val_acc)
split_val_loss_history.append(val_loss)
split_val_precision_history.append(val_precision)
split_val_recall_history.append(val_recall)
split_val_f1_history.append(val_f1)

# print(f"Time taken for fold {i}; lr: {LEARNING_RATE}; batch_size: {BATCH_SIZE}: {time.time() - start_fold_time}")

if early_stopped_epoch < NUM_EPOCHS:
    roc_auc, pr_auc = plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, early_stopped_epoch, val_preds, val_targets)
else:
    roc_auc, pr_auc = plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, NUM_EPOCHS, val_preds, val_targets)
historyDict = {
    'folds': 1,
    'train_transform': transforms,
    'train_accuracy': train_acc,
    'train_loss': train_loss,
    'val_accuracy': val_acc,
    'val_loss': val_loss,
    'val_precision': val_precision,
    'val_recall': val_recall,
    'f1_score': val_f1,
    'roc_auc': roc_auc,
    'prroc_auc': pr_auc
}

log_file.log(start_fold_time, historyDict)
    
if val_loss < best_metric:
    best_metric = val_loss
    best_batchsize = BATCH_SIZE
    best_learningrate = LEARNING_RATE

generateHeatMap(sample_images, sample_titles, model, DEVICE)
average_train_acc = np.mean(split_train_accuracy_history)
average_train_loss = np.mean(split_train_loss_history)
average_val_acc = np.mean(split_val_accuracy_history)
average_val_loss = np.mean(split_val_loss_history)
average_precision = np.mean(split_val_precision_history)
average_recall = np.mean(split_val_recall_history)
average_f1 = np.mean(split_val_f1_history)

finalHistoryDict = {
    'folds': None,
    'train_transform': transforms,
    'train_accuracy': average_train_acc,
    'train_loss': average_train_loss,
    'val_accuracy': average_val_acc,
    'val_loss': average_val_loss,
    'val_precision': average_precision,
    'val_recall': average_recall,
    'val_f1': average_f1,
    'roc_auc': roc_auc,
    'prroc_auc': pr_auc
}

log_file.log(start_time, finalHistoryDict)
print(f"\n{finalHistoryDict}")
print("-"*50)
# print(f"Best LR: {best_learningrate} with {best_metric} validation loss")
# print(f"Best BATCH_SIZE: {best_batchsize} with {best_metric} validation loss")