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
        # imageArr = self.stack_jpeg(imageArr)

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
        image = Image.open(jpeg_path).convert("RGB")
        # image = np.array(image)
        
        
        # Preprocessing
        # image = self.removeArtifacts(image)
        # image = self.apply_clahe(image)
        # image = self.apply_combined_filter(image)
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
    
    def stack_jpeg(self, image):
        stacked_image = np.stack((image,) * 3, axis=-1)
        return stacked_image
    
class CBISCombinedDataset(Dataset):
    def __init__(self, images, label, transform = None):
        self.images = images
        self.labels = label
        self.transform = transform
        
        # for i in range(len(self.dataframe)):
        #     row = self.dataframe.iloc[i]
        #     folder_name = row['folder_name']
        #     label = row['class_label']
        #     image_path = os.path.join(self.directory, str(i + 1), "1-1.dcm_downsized-cropped.jpeg")
        #     self.data.append(image_path)
        #     self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = self.load_jpeg(img)
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def load_jpeg(self, jpeg_path):
        image = Image.open(jpeg_path).convert("RGB")
        return image

class CBISNewDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/CBIS-DDSM_new", mode = 'train', transform = None):
        super(CBISNewDataset, self).__init__(directory, mode, transform)
        self.directory = os.path.join(directory, "calc_" + self.mode)
        if mode == "train":
            self.df_dir = os.path.join(directory, "calc_case_description_train_set.csv")
        else:
            self.df_dir = os.path.join(directory, "calc_case_description_test_set.csv")
            
        self.dataframe = pd.read_csv(self.df_dir)
        self.data = []
        self.labels = []
        for i in range(len(self.dataframe)):
            row = self.dataframe.iloc[i]
            folder_name = str(row['image file path']).split("/")[-2]
            # label = row['class_label']
            label = self.getLabel(row['pathology'])
            image_path = os.path.join(self.directory, str(i + 1), "1-1.jpeg")
            self.data.append(image_path)
            self.labels.append(label)
            

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        image_path = self.data[idx]
        label = self.labels[idx]
        
        
        # Dicom
        # image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm")
        # imageArr = self.load_dicom(image_path)
        
        # JPEG
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label
    
    def getLabel(self, label_text):
        label_text = label_text.lower()
        if "malignant" in label_text:
            return 1
        else:
            return 0


class CBISDataset(CustomImageDataset):
    def __init__(self, view: str = None, directory = "/home/emok/sq58/Code/Data/CBIS-DDSM", mode = 'train', transform = None):
        super(CBISDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        self.data = []
        self.labels = []
        if mode == "combined":
            for i in range(len(self.dataframe)):
                row = self.dataframe.iloc[i]
                if view is None or row['image view'] == view.upper():
                    folder_name = row['folder_name']
                    label = row['class_label']
                    image_path = os.path.join(self.directory, self.mode, str(i + 1), "1-1.dcm_downsized-cropped.jpeg")
                    self.data.append(image_path)
                    self.labels.append(label)
                
                    
        else:
            for i in range(len(self.dataframe)):
                row = self.dataframe.iloc[i]
                folder_name = row['folder_name']
                label = row['class_label']
                image_path = os.path.join(self.directory, self.mode, str(i + 1), "1-1.dcm_final.jpeg")
                self.data.append(image_path)
                self.labels.append(label)

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        image_path = self.data[idx]
        label = self.labels[idx]
        
        
        # Dicom
        # image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm")
        # imageArr = self.load_dicom(image_path)
        
        # JPEG
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            imageArr = self.transform(imageArr)

        return imageArr, label

class RSNADataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/RSNA", mode = 'train', transform = None):
        super(RSNADataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        self.data = []
        self.labels = []
        
        for i in range(len(self.dataframe)):
            row = self.dataframe.iloc[i]
            folder_name = row['patient_id']
            dcm_file = row['image_id']
            label = row['cancer']
            image_path = os.path.join(self.directory, self.mode, folder_name, dcm_file + ".dcm")
            self.data.append(image_path)
            self.labels.append(label)
        

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        image_path = self.data[idx]
        label = self.labels[idx]

        # Load the image using PIL
        imageArr = self.load_dicom(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    

class VinDrDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/VinDr", mode="train", transform = None):
        super(VinDrDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        self.data = []
        self.labels = []
        
        for i in range(len(self.dataframe)):
            row = self.dataframe.iloc[i]
            folder_name = row['folder_name']
            dcm_file = row['image_id']
            label = row['cancer']
            image_path = os.path.join(self.directory, self.mode, folder_name, dcm_file + ".dicom")
            self.data.append(image_path)
            self.label.append(label)
        

    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        image_path = self.data[idx]
        label = self.labels[idx]

        # Load the image using PIL
        imageArr = self.load_dicom(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    

class CMMDDataset(CustomImageDataset):
    def __init__(self, directory = "/home/emok/sq58/Code/Data/CMMD", mode = "train", transform=None):
        super(CMMDDataset, self).__init__(directory, mode, transform)
        self.dataframe = pd.read_csv(self.df_dir)
        self.data = []
        self.labels = []
        
        for i in range(len(self.dataframe)):
            row = self.dataframe.iloc[i]
            folder_name = row['folder_name']
            label = row['class_label']
            image_path = os.path.join(self.directory, self.mode, str(i + 1), "1-1.dcm_final.jpeg")
            self.data.append(image_path)
            self.labels.append(label)

    
    def __getitem__(self, idx):
        # Load and preprocess data sample at index idx
        image_path = self.data[idx]
        label = self.labels[idx]
        
        # # Dicom
        # image_path = os.path.join(self.directory, self.mode, str(idx + 1), "1-1.dcm")
        # # Load the image using PIL
        # imageArr = self.load_dicom(image_path)
        
        # JPEG
        imageArr = self.load_jpeg(image_path)

        if self.transform:
            image = self.transform(imageArr)

        return image, label
    
