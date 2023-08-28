import numpy as np

def normalize_intensity(pixel_data):
    min_intensity = np.min(pixel_data)
    max_intensity = np.max(pixel_data)
    normalized_data = (pixel_data - min_intensity) / (max_intensity - min_intensity)
    return normalized_data