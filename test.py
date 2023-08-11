import pandas as pd
import os
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution
import time

dataframe_directory = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/Data/CMMD/cleaned_clinicaldata.csv'
df = pd.read_csv(dataframe_directory)

data_folder = '/Users/enoch/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/CNYang/OFYP/Coding/Data/CMMD/dataset/CMMD/'
all_labels_1 = []
all_images_1 = []
all_labels_2 = []
all_images_2 = []
RESAMPLE_RESOLUTION = (224, 224)

algo1_start = time.time_ns()
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

                image_label = df.loc[df['ID1'] == ID1, 'class_label']
                all_labels_1.append(image_label)
end_time = time.time_ns()
algo1_time = end_time - algo1_start

algo2_start = time.time_ns()
# Algo 2
for index, patient in enumerate(df['ID1']):
    curr_folder = os.path.join(data_folder, patient)
    # image_label = df.loc[df['ID1'] == patient, 'class_label']
    image_label = df.iloc[[index]]['class_label']

    for root, dirs, files in os.walk(curr_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                curr_file_path = os.path.join(root, file)
                print(curr_file_path)

                pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
                normalized_data = normalize_intensity(pixel_data)
                resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
                all_images_2.append(resampled_image)
                all_labels_2.append(image_label)
end_time = time.time_ns()
algo2_time = end_time - algo2_start


print(f"Algo1_time = {algo1_time}")
print(len(all_labels_1))
print(len(all_images_1))
print()
print(f"Algo2_time = {algo2_time}")
print(len(all_labels_2))
print(len(all_images_2))

print(f"Are they equal? \nAll_labels = {all_labels_1 == all_labels_2} \nAll_images = {all_images_2 == all_images_1}")