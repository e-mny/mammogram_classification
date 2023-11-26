#!/bin/bash -l
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Kai Xi,  Feb 2015
#              help@massive.org.au

# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=CBISresnet_mammo


# To set a project account for credit charging, 
#SBATCH --account=sq58


# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Request for GPU, 
#
# Option 1: Choose any GPU whatever m2070 or K20
# Note in most cases, 'gpu:N' should match '--ntasks=N'

#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

# SBATCH --gres=gpu:P100:1
# SBATCH --partition=m3h

# SBATCH --gres=gpu:T4:1
# SBATCH --partition=gpu

# SBATCH --gres=gpu:A100:1
# SBATCH --partition=gpu


# Memory usage (MB)
#SBATCH --mem-per-cpu=16000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=02-23:00:00
# SBATCH --time=04-23


# To receive an email when job completes or fails
#SBATCH --mail-user=enoch.mok@monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the file for output (stdout)
#SBATCH --output=./sbatchlog/resnet/CBISresnet_mammo-%j.out

# Set the file for error log (stderr)
#SBATCH --error=./sbatchlog/resnet/CBISresnet_mammo-%j.err


# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name


# Command to run a gpu job
# For example:
# module load cuda/7.0
# nvidia-smi
# deviceQuery

EPOCHS="$1"
DATA_AUG="$2"

# activate conda
# module load conda
eval "$(conda shell.bash hook)"
conda activate exp1
# srun --exclusive python main.py --model resnet18 --dataset CBIS-DDSM
# srun --exclusive python main.py --model resnet34 --dataset CBIS-DDSM

if [ "$DATA_AUG" == "true" ]; then
    echo "Running script with data augmentation"
    srun --exclusive python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs $EPOCHS --data_augment --early_stopping
else
    echo "Running script without data augmentation"
    srun --exclusive python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs $EPOCHS --no-data_augment
fi