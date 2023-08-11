import pandas as pd
import os
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution
# import time
import pickle
import numpy as np

dataframe_directory = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/Data/CMMD/cleaned_clinicaldata.csv'
df = pd.read_csv(dataframe_directory)

data_folder = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/Data/CMMD/dataset/CMMD/'
all_labels_1 = []
all_images_1 = []
RESAMPLE_RESOLUTION = (224, 224)

# Algo 1
for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                curr_file_path = os.path.join(root, file)
                pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
                normalized_data = normalize_intensity(pixel_data)
                resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
                all_images_1.append(resampled_image)


                # Find corresponding label
                # uid_instance_id = root.split("/")[-1]
                # image_label = df.loc[df['uid_instance_id'] == uid_instance_id, 'pathology']
                # print(f"uid_instance_id: {uid_instance_id}")
                # print(f"image_label: {image_label}")

                # exists = df['uid_instance_id'].isin([uid_instance_id]).any()

                # if exists:
                #     print(f"{uid_instance_id} exists in the column.")
                # else:
                #     print(f"{uid_instance_id} does not exist in the column.")

                # Find corresponding label (CMMD)
                ID1 = root.split("/")[-3]
                # print(ID1)
                # exists = df['ID1'].isin([ID1]).any()

                # if exists:
                #     print(f"{ID1} exists in the column.")
                # else:
                #     print(f"{ID1} does not exist in the column.")

                # image_label = df.loc[df['ID1'] == ID1, 'class_label']

                index = df.index[df['ID1'] == ID1].tolist()[0]

                # Get the corresponding value from ColumnB
                image_label = df.at[index, 'class_label']
                all_labels_1.append(image_label)

# print(all_images_1)
print(all_labels_1)
print(len(all_images_1))
print(len(all_labels_1))
print(type(all_labels_1[0]))


images_pickle = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/experiment1/image_data.pickle'
labels_pickle = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/experiment1/label_data.pickle'

# Save the list to the pickle file
with open(images_pickle, 'wb') as pickle_file:
    pickle.dump(np.array(all_images_1), pickle_file)

with open(labels_pickle, 'wb') as pickle_file:
    pickle.dump(np.array(all_labels_1), pickle_file)
    