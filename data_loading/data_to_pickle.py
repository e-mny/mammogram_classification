import os
import pickle
import numpy as np
import pandas as pd
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution

def createDataList(home_dir, RESAMPLE_RESOLUTION):
    all_images = []
    all_labels = []
    images_pickle = os.path.join(home_dir, 'image_data.pickle')
    labels_pickle = os.path.join(home_dir, 'label_data.pickle')
    dataframe_directory = os.path.join(home_dir, "train.csv")
    data_folder = os.path.join(home_dir, "train_images")
    df = pd.read_csv(dataframe_directory)

    for i in range(len(df)):
        if ((i % 10) == 0):
            print(i)
        # print(i)
        # print(f"Index: {i}, folder_name: {df['folder_name'].iloc[i]}, pathology: {df['pathology'].iloc[i]}, class_label: {df['class_label'].iloc[i]}")
        curr_patient_folder = os.path.join(data_folder, str(df['patient_id'].iloc[i]))
        curr_img_id = os.path.join(curr_patient_folder, str(df['image_id'].iloc[i]))
        curr_label = df['cancer'].iloc[i]
        
        # for root, _, files in os.walk(curr_patient_folder):
        #     # print(_)
        #     # for file in files:
        #     curr_file_path = os.path.join(root, curr_img_id + ".dcm")
        #     # print(curr_file_path)
        #     # if curr_file_path.lower().endswith('.dcm'): # Sanity check
        #     pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
        #     normalized_data = normalize_intensity(pixel_data)
        #     resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
        #     all_images.append(resampled_image)
        #     all_labels.append(curr_label)

        # curr_file_path = os.path.join(curr_patient_folder, curr_img_id + ".dcm")
        curr_file_path = curr_img_id + ".dcm"
        print(curr_file_path)
        # print(curr_file_path)
        # if curr_file_path.lower().endswith('.dcm'): # Sanity check
        pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
        normalized_data = normalize_intensity(pixel_data)
        resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
        all_images.append(resampled_image)
        all_labels.append(curr_label)

    print("all_images & all_labels appended")
    # Save the list to the pickle file
    with open(images_pickle, 'wb') as pickle_file:
        pickle.dump(np.array(all_images), pickle_file)

    with open(labels_pickle, 'wb') as pickle_file:
        pickle.dump(np.array(all_labels), pickle_file)

    return images_pickle, labels_pickle
