import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import numpy as np


def plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, num_epochs, y_preds, y_targets):
    print("Plotting graph now")
    timenow = datetime.datetime.now()
    formatted_datetime = timenow.strftime("%m-%d-%Y_%H%M%S")
    f1_scores = []

    # AUC
    fpr, tpr, _ = roc_curve(y_preds, y_targets)
    roc_auc = auc(fpr, tpr)
    print(f"ROC_AUC: {roc_auc}")
    # print(f"FPR: {fpr}, TPR: {tpr}")
    # print(f"Predicted: {y_preds}")
    # print(f"Ground truth: {y_targets}")

    # PRAUC
    precision, recall, thresholds = precision_recall_curve(y_preds, y_targets)
    pr_auc = auc(recall, precision)
    print(f"PRROC_AUC: {pr_auc}")

    # Accuracy and Losses

    # F1 Score
    f1_scores = 2 * (precision * recall) / (precision + recall)


    # Create subplots
    # Plot the F1 score curve
    plt.figure()
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', color='b')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(f"./results/f1_score/{formatted_datetime}_{DATASET}_{MODEL}.png")
    plt.clf() # Clear figure
    # plt.show()

    # fig, ax = plt.subplots(3, 1, figsize=(15,20))
    # fig.set_figure = (6, 30)
    # ax1, ax2, ax3 = ax

    # Plot ROC AUC curve and PR AUC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(recall, precision, color='blue', lw=2, label='PR AUC (area = %0.2f)' % pr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.autoscale(axis='y')
    plt.xlabel('False Positive Rate / Recall')
    plt.ylabel('True Positive Rate / Precision')
    plt.title('ROC and PR AUC Curves')
    plt.legend(loc="lower right")
    plt.savefig(f"./results/roc/{formatted_datetime}_{DATASET}_{MODEL}.png")
    plt.clf() # Clear figure

    # Plot accuracy and loss
    # ax2.subplot(1, 2, 1)
    # plt.figure()
    plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss_history, label='Validation Loss')
    plt.ylim([0.0, 1.4])
    # plt.autoscale(axis='y')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc="upper left")
    plt.savefig(f"./results/loss/{formatted_datetime}_{DATASET}_{MODEL}.png")
    plt.clf() # Clear figure

    # ax2.subplot(1, 2, 2)
    # plt.figure()
    plt.plot(range(1, num_epochs+1), train_accuracy_history, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracy_history, label='Validation Accuracy')
    plt.ylim([0.4, 1])
    # plt.autoscale(axis='y')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc="upper left")
    plt.savefig(f"./results/accuracy/{formatted_datetime}_{DATASET}_{MODEL}.png")

    return roc_auc, pr_auc