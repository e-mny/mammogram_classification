import pandas as pd
import os
from torch.utils.data import Dataset, random_split
import pydicom
import numpy as np
from PIL import Image
import torch
import cv2
from data_loading.preprocessFunc import removeArtifacts

class CustomImageDataset(Dataset):
    def __init__(self, directory, mode, transform=None):
        self.directory = directory
        self.mode = mode
        if self.mode == "train":
            self.df_dir = os.path.join(self.directory, "train.csv")
        elif self.mode == "val":
            self.df_dir = os.path.join(self.directory, "val.csv")
        elif self.mode == "test":
            self.df_dir = os.path.join(self.directory, "test.csv")
        else:
            self.df_dir = os.path.join(self.directory, "combined_data.csv")
            
        self.transform = transform
        self.removeArtifacts = removeArtifacts
       
        # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        # # Load the image using PIL
        # imageArr = self.load_dicom(image_path)
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    
    def load_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        # Convert to 8-bit grayscale
        image = image - np.min(image)
        image = (image / np.max(image) * 255).astype(np.uint8)

        imagearray = Image.fromarray(image.astype(float))
        return imagearray
    
    def load_jpeg(self, jpeg_path):
        image = Image.open(jpeg_path)
        image = np.array(image)
        
        # Preprocessing
        # image = self.removeArtifacts(image)
        # image = self.apply_clahe(image)
        image = self.apply_combined_filter(image)
        # image = self.toPILImage(image)
        return image
    
    def apply_clahe(self, image):
        # Apply CLAHE to the image
        clahe_image = self.clahe.apply(image)
        return clahe_image

    def apply_mean_filter(self, image):
        return cv2.blur(image, (5, 5))

    def apply_gaussian_filter(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def apply_median_filter(self, image):
        return cv2.medianBlur(image, 5)
    
    def apply_combined_filter(self, image):
        return cv2.addWeighted(self.apply_mean_filter(image), 0.5, self.apply_median_filter(image), 0.5, 0)
    
    def toPILImage(self, image):
        return Image.fromarray(image)
    
    
    


class CBISDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/CBIS-DDSM", mode = 'train', transform = None):
        super(CBISDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        row = self.dataframe.iloc[idx]
        folder_name = row['folder_name']
        label = row['class_label']
        # Dicom
        # image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm")
        # imageArr = self.load_dicom(image_path)
        
        # JPEG
        image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm.jpeg")
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label

class RSNADataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/RSNA", mode = 'train', transform = None):
        super(RSNADataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        row = self.dataframe.iloc[idx]
        folder_name = row['patient_id']
        dcm_file = row['image_id']
        label = row['cancer']
        image_path = os.path.join(self.directory, self.mode, folder_name, dcm_file + ".dcm")

        # Load the image using PIL
        imageArr = self.load_dicom(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    

class VinDrDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/VinDr", mode="train", transform = None):
        super(VinDrDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        row = self.dataframe.iloc[idx]
        folder_name = row['study_id']
        dcm_file = row['image_id']
        label = row['cancer']
        image_path = os.path.join(self.directory, self.mode, folder_name, dcm_file + ".dicom")

        # Load the image using PIL
        imageArr = self.load_dicom(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    

class CMMDDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/CMMD", mode = "train", transform=None):
        super(CMMDDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)

        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)
    
    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        row = self.dataframe.iloc[idx]
        folder_name = row['ID1']
        label = row['class_label']
        
        # # Dicom
        # image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm")
        # # Load the image using PIL
        # imageArr = self.load_dicom(image_path)
        
        # JPEG
        image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm.jpeg")
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    
