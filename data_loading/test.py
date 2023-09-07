from data_to_pickle import createDataList

RESAMPLE_RESOLUTION = (224, 224)
home_dir = '/home/emok/sq58_scratch/emok/Data/CMMD/'

_, _ = createDataList(home_dir = home_dir, RESAMPLE_RESOLUTION = RESAMPLE_RESOLUTION)