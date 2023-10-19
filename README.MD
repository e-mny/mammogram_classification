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

---


<h3>Key Files</h3>

Refer to /models/modelFactory.py for:
- available models 
- edit the classifier layer 
- control which layers to freeze

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

Datasets directories:
1) CBIS-DDSM Dataset - '/home/emok/sq58/Code/Data/CBIS-DDSM/combined'
2) CMMD Dataset - '/home/emok/sq58_scratch/emok/Data/CMMD/images'
3) RSNA Dataset - '/home/emok/sq58_scratch/emok/Data/RSNA/train_images'
4) VinDr Dataset - '/home/emok/sq58_scratch/emok/Data/VinDr/images'