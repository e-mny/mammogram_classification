from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def displayMetrics(y_targets, y_preds):
    metrics = {
            "Train Accuracy": train_accuracy,
            "Train Loss": train_loss_history[-1],
            "Validation Accuracy": val_accuracy_score,
            "Validation Loss": val_loss_history[-1],
            "Validation Precision": val_precision_score,
            "Validation Recall": val_recall,
            "Validation F1": val_f1
        }

    val_accuracy_score = accuracy_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds, average='weighted')
    val_recall = recall_score(val_targets, val_preds, average='weighted')
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    roc_auc = roc_auc_score(y_targets, y_preds)
    

    print("-" * 10)
    print("Final Metrics")
    print(classification_report(y_targets, y_preds, target_names = ["Benign", "Malignant"]))
    print(f"ROC Score: {roc_auc}")
