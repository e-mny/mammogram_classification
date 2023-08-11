import torch 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def train(model, train_loader, val_loader, device, criterion, optimizer, epochs):
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    print("Starting training now")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            # inputs = inputs.unsqueeze(1)
            # labels = labels.unsqueeze(1)
            # print(inputs.shape)
            # print(torch.unique(labels))
            model.to(device)
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = correct_train / total_train
        train_loss_history.append(train_loss / len(train_loader))
        train_accuracy_history.append(train_accuracy)
        
        # Validation
        print("Starting Validation")
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # model.to(device)
                # inputs = inputs.unsqueeze(1)
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
            
        val_accuracy = correct_val / total_val
        val_loss_history.append(val_loss / len(val_loader))
        val_accuracy_history.append(val_accuracy)
        val_accuracy_score = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_confusion = confusion_matrix(val_targets, val_preds)
    
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {train_loss_history[-1]:.4f}, Train Accuracy: {train_accuracy:.4f} - "
            f"Validation Loss: {val_loss_history[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            f"Validation Accuracy: {val_accuracy_score:.4f}, "
            f"Validation Precision: {val_precision:.4f}, "
            f"Validation Recall: {val_recall:.4f}, "
            f"Validation F1-score: {val_f1:.4f}")


    return train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_preds, val_targets