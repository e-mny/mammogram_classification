#!/bin/bash

DATASET="cbis"

# First argument = EPOCHS
# Second argument = DataAugmentation Boolean (true/false)

# Job 1
sbatch $DATASET-resnet-slurm-job 100 false

# Job 2
# sbatch $DATASET-resnet-slurm-job 100 false

# Job 3
# sbatch $DATASET-resnet-slurm-job 10 false

# # Submit job 1
sbatch $DATASET-dense-slurm-job 100 false

# # Submit job 2
sbatch $DATASET-efficient-slurm-job 100 false

# # Submit job 3
sbatch $DATASET-mobile-slurm-job 100 false

# # Submit job 4
sbatch $DATASET-resnet-slurm-job 100 false

# # Submit job 5
sbatch $DATASET-xception-slurm-job 100 false

# # Submit job 6
sbatch $DATASET-resnext-slurm-job 100 false
