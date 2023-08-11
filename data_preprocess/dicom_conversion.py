import os
import numpy as np
import pydicom



def load_and_preprocess_dicom(dicom_path):
    # Load DICOM file
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Extract pixel data and rescale to Hounsfield units (if necessary)
    pixel_data = dicom_data.pixel_array.astype(np.float32)
    if 'RescaleSlope' in dicom_data:
        pixel_data = pixel_data * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
    
    return pixel_data, dicom_data


