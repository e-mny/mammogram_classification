## Breast Cancer Classification Baseline Model

1) Implement classification model 
2) Implement classification model into FL framework
3) Test various experiments to see whether FL helps or not


### Properties

Trying on CBIS-DDSM dataset
- 3568 images (2111 Benign; 1457 Malignant)

Adam Optimizer\
1e-4 learning rate\
Mini-batch size: 32\
Resampled resolution: 224 x 224\
Softmax cross-entropy loss\

### Running the code

To run the code:
`python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs 50 --data_augment --early_stopping`

This will run the code using ResNet50 model with the CBIS-DDSM dataset with the following parameters:
- 50 epochs
- Data Augmentation
- Early Stopping (Patience: 5)

---

Refer to /models/modelFactory.py for the available models that can be used