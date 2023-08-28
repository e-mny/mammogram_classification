from skimage.transform import resize

def resample_to_resolution(image, dicom_data, new_resolution):
    # if dicom_data.PixelSpacing:
    #     spacing = dicom_data.PixelSpacing
    # resample_factor = [spacing[0] / new_resolution[0], spacing[1] / new_resolution[1]]
    
    resampled_image = resize(image, output_shape=new_resolution, order=3, preserve_range=True)
    
    return resampled_image

