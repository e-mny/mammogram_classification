import datetime
import time

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def calcTimeElapsed(self, start_time):
        timeTaken = datetime.timedelta(seconds=time.time() - start_time)
        timeTakenStr = str(timeTaken).split(".")[0]
        return timeTakenStr
    

    def getCurrTime(self):
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H%M") # 09-02-2023-1523
        return timestamp

    def log(self, start_time, dataDict):
        if dataDict['folds'] == None:
            fold_num = "Average of all 5 folds"
        else:
            fold_num = dataDict['folds']
        train_accuracy = dataDict['train_accuracy']
        train_loss = dataDict['train_loss']
        val_accuracy = dataDict['val_accuracy']
        val_loss = dataDict['val_loss']
        val_precision = dataDict['val_precision']
        val_recall = dataDict['val_recall']
        roc_auc = dataDict['roc_auc']
        pr_auc = dataDict['prroc_auc']

        message = (
            f"Fold Number: {fold_num}\n"
            f"Train Accuracy: {train_accuracy}\n"
            f"Train Loss: {train_loss}\n"
            f"Val Accuracy: {val_accuracy}\n"
            f"Val Loss: {val_loss}\n"
            f"Val Precision: {val_precision}\n"
            f"Val Recall: {val_recall}\n"      
            f"Val F1-Score: {self.calcF1Score(val_precision, val_recall)}\n"
            f"ROC: {roc_auc}\n"  
            f"PRROC: {pr_auc}\n"  
            )
        
        with open(self.log_file, 'a') as file:
            file.write("\n")
            file.write(self.getCurrTime())
            file.write("\n")
            file.write(f"Time taken: {self.calcTimeElapsed(start_time)}")
            file.write("\n")
            file.write(message)
            file.write("\n")
        # print(log_message, end='')
        
    def calcF1Score(self, precision, recall):
        return round((precision * recall * 2) / (precision + recall), 4)
    
    def calcAverage(self, history):
        return round(sum(history) / len(history), 4)
