# from data_to_pickle import createDataList

RESAMPLE_RESOLUTION = (224, 224)
home_dir = '/home/emok/sq58_scratch/emok/Data/RSNA/'

# _, _ = createDataList(home_dir = home_dir, RESAMPLE_RESOLUTION = RESAMPLE_RESOLUTION)

import pandas as pd
import concurrent.futures
import os
import multiprocessing
import pickle
import numpy as np
from data_preprocess.dicom_conversion import load_and_preprocess_dicom
from data_preprocess.normalize_intensity import normalize_intensity
from data_preprocess.resample import resample_to_resolution

def process_chunk(chunk):
    # Process the chunk (you can modify this part)
    # print(chunk)
    chunk_images = []
    chunk_labels = []

    for i in range(len(chunk)):
        print(i)
        curr_patient_folder = os.path.join(data_folder, str(chunk['patient_id'].iloc[i]))
        curr_img_id = os.path.join(curr_patient_folder, str(chunk['image_id'].iloc[i]))
        curr_label = chunk['cancer'].iloc[i]
        # print(curr_label)

        curr_file_path = curr_img_id + ".dcm"
        # print(curr_file_path)
        # if curr_file_path.lower().endswith('.dcm'): # Sanity check
        pixel_data, dicom_data = load_and_preprocess_dicom(curr_file_path)
        normalized_data = normalize_intensity(pixel_data)
        resampled_image = resample_to_resolution(normalized_data, dicom_data, RESAMPLE_RESOLUTION)
        chunk_images.append(resampled_image)
        chunk_labels.append(curr_label)
        
    return chunk_images, chunk_labels

if __name__ == "__main__":
    all_images = []
    all_labels = []
    images_pickle = os.path.join(home_dir, 'image_data_parallel.pickle')
    labels_pickle = os.path.join(home_dir, 'label_data_parallel.pickle')
    dataframe_directory = os.path.join(home_dir, "train.csv")
    data_folder = os.path.join(home_dir, "train_images")
    df = pd.read_csv(dataframe_directory)

    num_threads = multiprocessing.cpu_count()  # Use all available CPU cores
    # chunk_size = len(df) // num_threads  # Adjust the chunk size as needed
    print(f"NumThreads = {num_threads}")
    # print(f"ChunkSize = {chunk_size}")

    # Split the DataFrame into chunks for parallel processing
    chunks = np.array_split(df, num_threads)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     for result1, result2 in executor.map(process_chunk, chunks):
    #         all_images.append(result1)
    #         all_labels.append(result2)
    #         print(f"Len(all_images) = {len(all_images)}")

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_threads)

    # Process the chunks in parallel
    chunk_images, chunk_labels = pool.map(process_chunk, chunks)

    pool.close()
    pool.join()

    # Combine the processed chunks back into a single DataFrame
    # processed_df = pd.concat(processed_chunks)
    print(len(chunk_images))
    print(len(chunk_labels))

    # Dump the processed results into separate pickle files
    with open(images_pickle, "wb") as f:
        pickle.dump(all_images, f)

    with open(label_pickle, "wb") as f:
        pickle.dump(all_labels, f)
