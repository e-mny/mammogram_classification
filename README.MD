# Federated Learning for Breast Cancer Classification

This project focuses on implementing and evaluating classification models for breast cancer diagnosis using Federated Learning (FL).
Specifically, the project aims to:
1) Implement a classification model for mammograms.
2) Incorporate the classification model into a Federated Learning framework.
3) Conduct experiments to assess the impact of Federated Learning on classification performance.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview
The project aims to classify mammograms using Deep Learning. It uses Convolutional Neural Networks (CNNs) to analyse medical images and determine whether the lesion is malignant or benign.

## Installation
To install the required libraries, use the following commands:

```bash
pip install -r requirements.txt
```

## Usage
To execute training using ResNet50 on the CBIS-DDSM dataset, use the following command:

```bash
python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs 100 --data_augment --early_stopping
```

This command runs the code with the following parameters:

Model: ResNet50
Dataset: CBIS-DDSM
Epochs: 100
With Data Augmentation
Early Stopping (Patience: 5)

If using SLURM, use:

```bash
sbatch cbis-resnet-slurm-job 100 true
```
First argument (100) is the number of epochs
Second argument (true) indicates data augmentation

## Dataset
Refer to [Datasets.MD](data/datasets.MD).

For data augmentation properties, refer to [data_loader.py](data_loading/data_loader.py).

## Model Architecture
For our experiments, we used ResNet50 for training as we found it to have the best performance out of the state-of-the-art (SOTA) networks (Xception, MobileNet_v2, DenseNet121, EfficientNet_b0).

To test this code using other networks, refer to [modelFactory.py](models/modelFactory.py) for
- available models
- editing of classifier layer
- controlling which layers to freeze

## Training
The model was trained using a batch size of 512 for 100 epochs with a learning rate of 0.001. We utilized transfer learning with a pre-trained ResNet50 model as the base and fine-tuned it on our dataset.

Training file can be found [train_loader.py](train/train_loader.py).


## Evaluation
The model achieved an accuracy of __ on the test set, outperforming previous state-of-the-art models. Evaluation metrics included accuracy, precision, recall, and F1-score.

Accuracy:
Precision:
Recall:
F1-Score:

## Results
Results are saved in the [Results](results) folder, formatted as MM-DD-YYYY_HHMMSS_{dataset}_{model}.jpg.

Our studies showed that ...