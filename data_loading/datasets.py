import pandas as pd
import os
from torch.utils.data import Dataset, random_split
import pydicom
import numpy as np
from PIL import Image
import torch


class CBISDataset(Dataset):
    def __init__(self, transform=None):
        self.data_dir = "/home/emok/sq58_scratch/emok/Data/CBIS-DDSM"
        self.transform = transform
        self.df_dir = os.path.join(self.data_dir, "whole_mammo_images_data.csv")
        self.df = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        folder_name = self.df[['folder_name']].iloc[idx].item()
        image_path = os.path.join(self.data_dir, "images", folder_name, "1-1.dcm")
        label = self.df[['class_label']].iloc[idx].item()
        imageArr = self.load_dicom(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label
    
    def load_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        # Convert to 8-bit grayscale
        image = image - np.min(image)
        image = (image / np.max(image) * 255).astype(np.uint8)

        imagearray = Image.fromarray(image)
        return imagearray
    
    def split(self, split_ratio=0.7, random_seed=None):
        # Determine the sizes of training and validation datasets
        num_samples = len(self)
        num_train = int(split_ratio * num_samples)
        num_val = num_samples - num_train

        # Use random_split to create training and validation datasets
        if random_seed is not None:
            torch.manual_seed(random_seed)

        train_dataset, val_dataset = random_split(
            self, [num_train, num_val]
        )

        return train_dataset, val_dataset

class RSNADataset(Dataset):
    def __init__(self, transform=None):
        self.data_dir = "/home/emok/sq58_scratch/emok/Data/RSNA"
        self.transform = transform
        self.df_dir = os.path.join(self.data_dir, "combined_data.csv")
        self.df = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        folder_name = self.df[['patient_id']].iloc[idx].item()
        dcm_file = self.df[['image_id']].iloc[idx].item()
        
        image_path = os.path.join(self.data_dir, "train_images", folder_name, dcm_file + ".dcm")
        label = self.df[['cancer']].iloc[idx].item()
        imageArr = self.load_dicom(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label
    
    def load_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        # Convert to 8-bit grayscale
        image = image - np.min(image)
        image = (image / np.max(image) * 255).astype(np.uint8)

        imagearray = Image.fromarray(image)
        return imagearray
    
    def split(self, split_ratio=0.7, random_seed=None):
        # Determine the sizes of training and validation datasets
        num_samples = len(self)
        num_train = int(split_ratio * num_samples)
        num_val = num_samples - num_train

        # Use random_split to create training and validation datasets
        if random_seed is not None:
            torch.manual_seed(random_seed)

        train_dataset, val_dataset = random_split(
            self, [num_train, num_val]
        )

        return train_dataset, val_dataset

class VinDrDataset(Dataset):
    def __init__(self, transform=None):
        self.data_dir = "/home/emok/sq58_scratch/emok/Data/VinDr"
        self.transform = transform
        self.df_dir = os.path.join(self.data_dir, "combined_data.csv")
        self.df = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        folder_name = self.df[['study_id']].iloc[idx].item()
        dcm_file = self.df[['image_id']].iloc[idx].item()
        
        image_path = os.path.join(self.data_dir, "train_images", folder_name, dcm_file + ".dicom")
        label = self.df[['cancer']].iloc[idx].item()
        imageArr = self.load_dicom(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label
    
    def load_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        # Convert to 8-bit grayscale
        image = image - np.min(image)
        image = (image / np.max(image) * 255).astype(np.uint8)

        imagearray = Image.fromarray(image)
        return imagearray
    
    def split(self, split_ratio=0.7, random_seed=None):
        # Determine the sizes of training and validation datasets
        num_samples = len(self)
        num_train = int(split_ratio * num_samples)
        num_val = num_samples - num_train

        # Use random_split to create training and validation datasets
        if random_seed is not None:
            torch.manual_seed(random_seed)

        train_dataset, val_dataset = random_split(
            self, [num_train, num_val]
        )

        return train_dataset, val_dataset

class CMMDDataset(Dataset):
    def __init__(self, transform=None):
        self.data_dir = "/home/emok/sq58_scratch/emok/Data/CMMD"
        self.transform = transform
        self.df_dir = os.path.join(self.data_dir, "combined_data.csv")
        self.df = pd.read_csv(self.df_dir)
        
        # # Pickle Method
        # self.images_pickle = os.path.join(self.data_dir, "image_data.pickle")
        # self.labels_pickle = os.path.join(self.data_dir, "label_data.pickle")
        # with open(self.images_pickle, 'rb') as pickle_file:
        #     self.img_array = pickle.load(pickle_file)

        # with open(self.label_pickle, 'rb') as pickle_file:
        #     self.df_labels = pickle.load(pickle_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        folder_name = self.df[['ID1']].iloc[idx].item()
        
        # TODO THERE ARE MORE THAN 1 DCM FILE PER PATIENT
        image_path = os.path.join(self.data_dir, "images", folder_name, ".dcm")
        label = self.df[['class_label']].iloc[idx].item()
        imageArr = self.load_dicom(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label
    
    def load_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array

        # Convert to 8-bit grayscale
        image = image - np.min(image)
        image = (image / np.max(image) * 255).astype(np.uint8)

        imagearray = Image.fromarray(image)
        return imagearray


    def split(self, split_ratio=0.7, random_seed=None):
            # Determine the sizes of training and validation datasets
            num_samples = len(self)
            num_train = int(split_ratio * num_samples)
            num_val = num_samples - num_train

            # Use random_split to create training and validation datasets
            if random_seed is not None:
                torch.manual_seed(random_seed)

            train_dataset, val_dataset = random_split(
                self, [num_train, num_val]
            )

            return train_dataset, val_dataset