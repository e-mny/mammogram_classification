import pickle
import numpy as np

# Path to the pickle file
pickle_file_path = './label_data.pickle'

# Load the list from the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_list = np.array(pickle.load(pickle_file))

print(type(loaded_list[0]))
