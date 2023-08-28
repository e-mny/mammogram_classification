from data_to_pickleVinDr import createDataList

RESAMPLE_RESOLUTION = (224, 224)
home_dir = '/home/emok/sq58_scratch/emok/Data/VinDr/'

_, _ = createDataList(home_dir = home_dir, RESAMPLE_RESOLUTION = RESAMPLE_RESOLUTION)