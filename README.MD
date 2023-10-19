## Breast Cancer Classification Baseline Model

1) Implement classification model 
2) Implement classification model into FL framework
3) Test various experiments to see whether FL helps or not


### Running the code

To run the code:
`python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs 50 --data_augment --early_stopping`

This will run the code using ResNet50 model with the CBIS-DDSM dataset with the following parameters:
- 50 epochs
- Data Augmentation
- Early Stopping (Patience: 5)
<br></br>

\* If using SLURM, use `sbatch cbis-resnet-slurm-job 100 true` to submit a job using ResNet50 for CBIS-DDSM dataset for 100 epochs and with data augmentation.
- First argument (`100`) is the number of epochs
- Second argument (`true`) is the boolean value for data augmentation

---


<h3>Key Files</h3>

Refer to /models/modelFactory.py for:
- available models 
- editing the classifier layer 
- controling which layers to freeze

<br></br>
Refer to /data_loading/datasets.py for:
- usage of image files

<br></br>
Refer to /data_loading/data_loader.py for:
- data augmentation
- data loading
- calculating class distribution

<br></br>
Refer to /train/train_loader.py for:
- training of model

---

<h3>Results</h3>
Results can be found in the /results folder, formatted in MM-DD-YYYY_HHMMSS_{dataset}_{model}.jpg

---

<h3>Datasets directories:</h3>
1) CBIS-DDSM Dataset - '/home/emok/sq58/Code/Data/CBIS-DDSM/combined'<br>
2) CMMD Dataset - '/home/emok/sq58_scratch/emok/Data/CMMD/images'<br>
3) RSNA Dataset - '/home/emok/sq58_scratch/emok/Data/RSNA/train_images'<br>
4) VinDr Dataset - '/home/emok/sq58_scratch/emok/Data/VinDr/images'