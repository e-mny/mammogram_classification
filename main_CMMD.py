# from model import DenseNet
import pickle
import torch
import pandas as pd
import os
from data_loading.data_loader import createDataLoaders
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
data_folder = '/home/emok/sq58_scratch/emok/Data/CMMD/CMMD/'
dataset_name = data_folder.split("/")[-2]
data_folder = os.path.join(data_folder, 'dataset/CMMD')
RESAMPLE_RESOLUTION = (224, 224)
TRAIN_RATIO = 0.7  # 70% for training
TEST_RATIO = 0.2   # 20% for testing
VAL_RATIO = 0.1    # 10% for validation

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")



dataframe_directory = os.path.join(data_folder, 'cleaned_clinicaldata.csv')
df = pd.read_csv(dataframe_directory)

# Algo 1 (run this if first time)

# for root, dirs, files in os.walk(data_folder):
#         for file in files:
#             if file.lower().endswith('.dcm'):
#                 curr_file_path = os.path.join(root, file)
#                 pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
#                 normalized_data = normalize_intensity(pixel_data)
#                 resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
#                 all_images.append(resampled_image)


#                 # Find corresponding label
#                 # uid_instance_id = root.split("/")[-1]
#                 # image_label = df.loc[df['uid_instance_id'] == uid_instance_id, 'pathology']
#                 # print(f"uid_instance_id: {uid_instance_id}")
#                 # print(f"image_label: {image_label}")

#                 # exists = df['uid_instance_id'].isin([uid_instance_id]).any()

#                 # if exists:
#                 #     print(f"{uid_instance_id} exists in the column.")
#                 # else:
#                 #     print(f"{uid_instance_id} does not exist in the column.")

#                 # Find corresponding label (CMMD)
#                 ID1 = root.split("/")[-3]
#                 # print(ID1)
#                 # exists = df['ID1'].isin([ID1]).any()

#                 # if exists:
#                 #     print(f"{ID1} exists in the column.")
#                 # else:
#                 #     print(f"{ID1} does not exist in the column.")

#                 index = df.index[df['ID1'] == ID1].tolist()[0]

#                 # Get the corresponding value from ColumnB
#                 image_label = df.at[index, 'class_label']
#                 all_labels.append(image_label)


# # Algo 3 (Pickle) (use this for quicker retrieval of data after Algo 1 was done before)

# Path to the pickle file
label_file_path = os.path.join(data_folder, 'label_data.pickle')
image_file_path = os.path.join(data_folder, 'image_data.pickle')

# # Load the list from the pickle file
with open(image_file_path, 'rb') as image_pickle_file:
    all_images = pickle.load(image_pickle_file)
with open(label_file_path, 'rb') as label_pickle_file:
    all_labels = pickle.load(label_pickle_file)


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
