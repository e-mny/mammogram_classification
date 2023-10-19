import torch 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from visualization.explainPred import generateHeatMap
import time
import numpy as np
from itertools import chain


def train(model, train_loader, val_loader, device, criterion, optimizer, epochs):
    PATIENCE = 5
    best_metric = float('inf')
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []
    print("Starting training now")
    for epoch in range(epochs):
        start_train_time = time.time()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_num, (inputs, labels) in enumerate(train_loader):
            # print("In Train_loader")
            # print(f"Current Batch: {batch_num}")
            # print(inputs, labels)
            # print(type(inputs), type(labels))
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            model = model.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            batch_num += 1
        
        train_accuracy_score = correct_train / total_train
        train_loss_score = train_loss / len(train_loader)
        train_accuracy_history.append(train_accuracy_score)
        train_loss_history.append(train_loss_score)
        
        print(f"Train time: {(time.time() - start_train_time):.2f}s")
        
        
        
        # Validation
        start_val_time = time.time()
        print("Starting Validation")
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                model = model.to(device)
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
            
        val_accuracy_score = correct_val / total_val
        val_loss_score = val_loss / len(val_loader)
        val_accuracy_history.append(val_accuracy_score)
        val_loss_history.append(val_loss_score)
        val_precision = precision_score(val_targets, val_preds, average='binary')
        val_precision_history.append(val_precision)
        val_recall = recall_score(val_targets, val_preds, average='binary')
        val_recall_history.append(val_recall)
        val_f1 = f1_score(val_targets, val_preds, average='binary')

        # Check if validation loss improved
        if val_loss_score < best_metric:
            best_metric = val_loss_score
            patience_counter = 0
            # Save model checkpoint if needed
            torch.save(model.state_dict(), './models/checkpoints/best_model.pth')
        else:
            patience_counter += 1
        
        # Check if early stopping criteria are met
        if patience_counter >= PATIENCE:
            print(f'Early stopping after {epoch} epochs.')
            break
        
        print(f"Val time: {(time.time() - start_val_time):.2f}s")
        print(f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {train_loss_history[-1]:.4f}, Train Accuracy: {train_accuracy_score:.4f} - "
            f"\nValidation Loss: {val_loss_history[-1]:.4f}, "
            f"Validation Accuracy: {val_accuracy_score:.4f}")
            # f"Validation Precision: {val_precision:.4f}, "
            # f"Validation Recall: {val_recall:.4f}, "
            # f"Validation F1-score: {val_f1:.4f}")

    # generateHeatMap(val_loader, model, device)
    
    

    return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets

def stratified_train(model, train_loader, val_loader, device, criterion, optimizer, epochs, early_stopping):
    PATIENCE = 5
    best_metric = float('inf')
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []
    # model = model.to(device)
    
    

    print("Starting training now")
    for epoch in range(epochs):
        start_train_time = time.time()
        model.train()
        # train_loss = 0.0
        # correct_train = 0
        # total_train = 0
        # for batch_num, (inputs, labels) in enumerate(train_loader):
        #     # If using CEL
        #     # labels = labels.type(torch.LongTensor)
        #     # inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #     # optimizer.zero_grad()
        #     # outputs = model(inputs)
        #     # loss = criterion(outputs, labels)
            
        #     # If using BCE Loss
        #     labels = labels.type(torch.FloatTensor)
        #     inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     # print()
        #     # print(inputs.shape)
        #     # print()
        #     # print(outputs)
        #     # print(outputs.shape)
        #     # print()
        #     # print(labels)
        #     # print(labels.shape)            
        #     loss = criterion(outputs.squeeze(), labels)


        #     loss.backward()
        #     optimizer.step()
            
        #     train_loss += loss.detach().item()
        #     _, predicted = torch.max(outputs.data, 1)
        #     total_train += labels.size(0)
        #     correct_train += (predicted == labels).sum().item()
        #     batch_num += 1
        
        # train_accuracy_score = correct_train / total_train
        # train_loss_score = train_loss / len(train_loader)
        # train_accuracy_history.append(train_accuracy_score)
        # train_loss_history.append(train_loss_score)
        
        # print(f"Train time: {(time.time() - start_train_time):.2f}s")
        
        
        
        # # Validation
        # start_val_time = time.time()
        # print("Starting Validation")
        # model.eval()
        # val_loss = 0.0
        # correct_val = 0
        # total_val = 0
        # val_preds = []
        # val_targets = []
        
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         # If using CEL
        #         # labels = labels.type(torch.LongTensor)
        #         # inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #         # outputs = model(inputs)
        #         # loss = criterion(outputs, labels)
                
        #         # If using BCE Loss
        #         labels = labels.type(torch.FloatTensor)
        #         print(labels)
        #         print()
        #         inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        #         outputs = model(inputs)
        #         print(outputs)
        #         print()
        #         loss = criterion(outputs.squeeze(), labels)
                
        #         val_loss += loss.detach().item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total_val += labels.size(0)
        #         correct_val += (predicted == labels).sum().item()
        #         val_preds.extend(predicted.detach().cpu().numpy())
        #         val_targets.extend(labels.detach().cpu().numpy().astype(int))
        # # print(val_preds) 
        # # print()  
        # # print(val_targets)   
        # val_accuracy_score = correct_val / total_val
        # val_loss_score = val_loss / len(val_loader)
        # val_accuracy_history.append(val_accuracy_score)
        # val_loss_history.append(val_loss_score)
        # val_precision = precision_score(val_targets, val_preds, average='binary')
        # val_precision_history.append(val_precision)
        # val_recall = recall_score(val_targets, val_preds, average='binary')
        # val_recall_history.append(val_recall)
        # val_f1 = f1_score(val_targets, val_preds, average='binary')

        # print(f"Val time: {(time.time() - start_val_time):.2f}s")
        # print(f"Epoch [{epoch+1}/{epochs}] - "
        #     f"Train Loss: {train_loss_history[-1]:.4f}, Train Accuracy: {train_accuracy_score:.4f} - "
        #     f"\nValidation Loss: {val_loss_history[-1]:.4f}, "
        #     f"Validation Accuracy: {val_accuracy_score:.4f}")
        
        # Check if validation loss improved
        # if val_loss_score < best_metric:
        #     best_metric = val_loss_score
        #     patience_counter = 0
        #     # Save model checkpoint if needed
        #     torch.save(model.state_dict(), './models/checkpoints/best_model.pth')
        # else:
        #     patience_counter += 1
            
        # # Check if early stopping criteria are met
        # if patience_counter >= PATIENCE:
        #     print(f'Early stopping after {epoch+1} epochs.')

        #     return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets, epoch+1
        
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            labels = labels.type(torch.LongTensor)
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
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                val_outputs = model(inputs)
                val_loss += criterion(val_outputs.squeeze(), labels).item()
                predicted = torch.round(val_outputs).squeeze()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_preds.append(predicted.detach().cpu().numpy())
                val_targets.append(labels.detach().cpu().numpy().astype(int))
                
        print(f"Val time: {(time.time() - start_val_time):.2f}s")
        val_loss /= len(val_loader)
        print(val_targets)
        print(val_preds)
        val_accuracy = correct_val / total_val
        val_precision = precision_score(val_targets, val_preds)
        val_precision_history.append(val_precision)
        val_recall = recall_score(val_targets, val_preds)
        val_recall_history.append(val_recall)
        val_f1 = f1_score(val_targets, val_preds)

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
                # Save model checkpoint if needed
            else:
                patience_counter += 1
                
            # Check if early stopping criteria are met
            if patience_counter >= PATIENCE:
                print(f'Early stopping after {epoch+1} epochs.')
                val_preds_flattened = list(chain.from_iterable(val_preds))
                val_targets_flattened = list(chain.from_iterable(val_targets))

                return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds_flattened, val_targets_flattened, epoch+1
        
    

    # Use itertools.chain to flatten the list
    val_preds_flattened = list(chain.from_iterable(val_preds))
    val_targets_flattened = list(chain.from_iterable(val_targets))
    
    # print(val_preds_flattened)
    # print(val_targets_flattened)
    
    return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds_flattened, val_targets_flattened, epochs
    
