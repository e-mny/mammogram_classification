import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
# from train.train_loader import stratified_train
from data_loading.data_loader import createDatasets
import os
from data_loading.datasets import CBISCombinedDataset
from models.modelFactory import replaceFCLayer

LEARNING_RATE = 1e-4
NUM_WORKERS = 6
print(f"NUM_WORKERS: {NUM_WORKERS}")
resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)  # Modify the output layer

# Define your ResNet-based classifier as a scikit-learn estimator.
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Replace the final classification layer to match the number of classes in your dataset
        # num_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(num_features, num_classes)
        self.resnet = replaceFCLayer(self.resnet, num_classes)

    def forward(self, x):
        return self.resnet(x)

X, y, transforms = createDatasets(dataset = "CBIS-DDSM", data_augment = True)
train_transforms, val_transforms = transforms
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = CBISCombinedDataset(X_train, y_train, transform=train_transforms)
val_dataset = CBISCombinedDataset(X_val, y_val, transform=val_transforms)  # No augmentation for validation
    
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=NUM_WORKERS, pin_memory=True)

# Define the pipeline with preprocessing and the ResNet classifier.
pipeline = Pipeline([
    ('clf', ResNetClassifier(num_classes=2))  # You can add preprocessing steps here.
])

param_grid = {
    'conv_layers': [2, 3, 4],
    'n_filters': [32, 64, 128],
    'filter_sizes': [(3, 3), (5, 5), (7, 7)],
    'fc_layers': [1, 2, 3, 4, 5, 6, 7],
    'fc_neurons': [128, 256, 512, 1024],
    'dropout': [0.3, 0.5],
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32, 64, 128, 256, 512],
    'epochs': [100, 200],
}


# Define a list of scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Create GridSearchCV with multiple scoring metrics
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    cv=5,  # Number of cross-validation folds
    refit="accuracy",
    verbose=2,
    n_jobs=-1
)

# Fit the grid search to your data
# grid_search.fit(X, y)

# Access the results for each scoring metric
# results = grid_search.cv_results_

grid_search.fit(X_train, y_train)  # You can pass the raw data to grid_search

best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print("Validation Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
results = grid_search.cv_results_
print("Results:", results)

# best_lr = best_params['lr']
# best_batch_size = best_params['batch_size']
# best_dropout = best_params['dropout']

# best_model = models.resnet50(pretrained=True)
# best_model.fc = nn.Linear(num_features, 1)  # Modify the output layer for binary classification

# criterion = nn.CrossEntropyLoss() # If using linear.out_features = 2
# optimizer = optim.Adam(best_model.parameters(), lr=LEARNING_RATE)

# stratified_train(best_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)  # Train with best hyperparameters
