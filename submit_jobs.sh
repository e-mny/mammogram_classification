#!/bin/bash

DATASET="cbis"

# First argument = EPOCHS
# Second argument = DataAugmentation Boolean (true/false)

# Job 1
sbatch $DATASET-resnet-slurm-job 200 true

# Job 2
sbatch $DATASET-resnet-slurm-job 200 false

# Job 3
# sbatch $DATASET-resnet-slurm-job 10 false

# # Submit job 1
# sbatch $DATASET-dense-slurm-job

# # Submit job 2
# sbatch $DATASET-efficient-slurm-job

# # Submit job 3
# sbatch $DATASET-mobile-slurm-job

# # Submit job 4
# sbatch $DATASET-resnet-slurm-job

# # Submit job 5
# sbatch $DATASET-xception-slurm-job

# # Submit job 6
# sbatch $DATASET-resnext-slurm-job
