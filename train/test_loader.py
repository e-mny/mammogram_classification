import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from train.train_loader import flattenList

def test(test_loader, model, device):
    # Testing the model
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    test_predicted = []
    test_targets = []
    
    # Define loss function
    # criterion = nn.CrossEntropyLoss() # If using linear.out_features = 2
    criterion = nn.BCELoss() # If using sigmoid
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            predicted = torch.round(outputs).squeeze()
            correct_test += (predicted.cpu().detach() == labels.cpu().detach()).sum().item()
            total_test += labels.size(0)
            test_predicted.append(predicted.cpu().detach().numpy().astype(int))
            test_targets.append(labels.cpu().detach().numpy().astype(int))

    accuracy = correct_test / total_test
    average_loss = test_loss / len(test_loader)

    test_targets, test_predicted = flattenList(test_targets), flattenList(test_predicted)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(test_targets, test_predicted)
    recall = recall_score(test_targets, test_predicted)
    f1 = f1_score(test_targets, test_predicted)

    # Calculate AUC and PRAUC
    auc_score = roc_auc_score(test_targets, test_predicted)
    precision_points, recall_points, _ = precision_recall_curve(test_targets, test_predicted)
    pr_auc = auc(recall_points, precision_points)

    print(f'Test Accuracy: {accuracy}')
    print(f'Average Test Loss: {average_loss}')
    print(f'Test Precision: {precision}')
    print(f'Test Recall: {recall}')
    print(f'Test F1 Score: {f1}')
    print(f'Test AUC: {auc_score}')
    print(f'Test PRAUC: {pr_auc}')
    
    return {"Accuracy": accuracy,
            "Loss": average_loss,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc_score,
            "PRAUC": pr_auc
            }