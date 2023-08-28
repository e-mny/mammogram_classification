import pickle
import numpy as np
import os

# Path to the pickle file
home_dir = '/home/emok/sq58_scratch/emok/Data/RSNA/'
image_pickle = os.path.join(home_dir, 'image_data.pickle')
label_pickle = os.path.join(home_dir, 'label_data.pickle')
# Load the list from the pickle file
with open(image_pickle, 'rb') as image_data:
    img_array = np.array(pickle.load(image_data))

with open(label_pickle, 'rb') as label_data:
    label_array = np.array(pickle.load(label_data))

print(f"img_array = {len(img_array)}")
print(f"label_array = {len(label_array)}")