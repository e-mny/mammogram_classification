import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from visualization.explainPred import generateHeatMap
import time
import numpy as np
from itertools import chain
from utils.config import PATIENCE, LEARNING_RATE, WEIGHT_DECAY

def train(model, train_loader, val_loader, device, epochs, early_stopping):
    best_metric = float('inf')
    end_epoch = None
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []
    
    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss() # If using linear.out_features = 2
    criterion = nn.BCELoss() # If using sigmoid
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    

    print("Starting training now")
    for epoch in range(epochs):
        start_train_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            labels = labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = torch.round(outputs).squeeze()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = correct_train / total_train
        train_loss /= len(train_loader)

        print(f"Train time: {(time.time() - start_train_time):.2f}s")

        # Validation
        start_val_time = time.time()
        print("Starting Validation")
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_predicted = []
        val_truth = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.type(torch.FloatTensor)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                val_outputs = model(inputs)
                val_loss += criterion(val_outputs.squeeze(), labels).item()
                predicted = torch.round(val_outputs).squeeze()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_predicted.append(predicted.detach().cpu().numpy().astype(int))
                val_truth.append(labels.detach().cpu().numpy().astype(int))
                
        print(f"Val time: {(time.time() - start_val_time):.2f}s")
        val_accuracy = correct_val / total_val
        val_loss /= len(val_loader)
        
        val_predicted_flattened = flattenList(val_predicted)
        val_truth_flattened = flattenList(val_truth)
        # print(val_truth_flattened)
        # print(val_predicted_flattened)
        val_precision = precision_score(val_truth_flattened, val_predicted_flattened)
        val_precision_history.append(val_precision)
        val_recall = recall_score(val_truth_flattened, val_predicted_flattened)
        val_recall_history.append(val_recall)
        val_f1 = f1_score(val_truth_flattened, val_predicted_flattened)

        # Print metrics
        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        print(f'Validation F1 Score: {val_f1:.4f}')
        print('-' * 50)

        train_accuracy_history.append(train_accuracy)
        train_loss_history.append(train_loss)
        val_accuracy_history.append(val_accuracy)
        val_loss_history.append(val_loss)
        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
    
        if early_stopping:
            # Check if validation loss improved
            if val_loss < best_metric:
                best_metric = val_loss
                patience_counter = 0
                
            else:
                patience_counter += 1
                
            # Check if early stopping criteria are met
            if patience_counter >= PATIENCE:
                print(f'Early stopping after {epoch+1} epochs.')
                # Save model checkpoint
                # torch.save(model.state_dict(), "./models/pretrained_CBISMassROI.pth")
                end_epoch = epoch+1
                break
        
    
    # val_predicted_flattened = flattenList(val_predicted)
    # val_truth_flattened = flattenList(val_truth)
    
    if not end_epoch: # No early stopping
        return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_predicted_flattened, val_truth_flattened, epochs
    else: 
        return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_predicted_flattened, val_truth_flattened, end_epoch
    
def flattenList(originallist):
    return list(chain.from_iterable(originallist))