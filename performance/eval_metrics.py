from performance.show_graph import plotGraph
from utils.config import NUM_EPOCHS, DATASET, MODEL

def evaluate_performance(train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, val_precision_history, val_recall_history, val_preds, val_targets, early_stopped_epoch):
    train_acc = train_accuracy_history[-1]
    train_loss = train_loss_history[-1]
    val_acc = val_accuracy_history[-1]
    val_loss = val_loss_history[-1]
    val_precision = val_precision_history[-1]
    val_recall = val_recall_history[-1]
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)

    return train_acc, train_loss, val_acc, val_loss, val_precision, val_recall, val_f1

def plot_and_log(train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, early_stopped_epoch, val_preds, val_targets):
    if early_stopped_epoch < NUM_EPOCHS:
        roc_auc, pr_auc = plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, early_stopped_epoch, val_preds, val_targets)
    else:
        roc_auc, pr_auc = plotGraph(DATASET, MODEL, train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history, NUM_EPOCHS, val_preds, val_targets)
    return roc_auc, pr_auc