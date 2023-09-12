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
        train_accuracy_history = dataDict['train_accuracy']
        train_loss_history = dataDict['train_loss']
        val_accuracy_history = dataDict['val_accuracy']
        val_loss_history = dataDict['val_loss']
        val_precision_history = dataDict['val_precision']
        val_recall_history = dataDict['val_recall']


        avgMsg = (
            f"Train Accuracy: {self.calcAverage(train_accuracy_history)}\n"
            f"Train Loss: {self.calcAverage(train_loss_history)}\n"
            f"Val Accuracy: {self.calcAverage(val_accuracy_history)}\n"
            f"Val Loss: {self.calcAverage(val_loss_history)}\n"
            f"Val Precision: {self.calcAverage(val_precision_history)}\n"
            f"Val Recall: {self.calcAverage(val_recall_history)}\n"      
            f"Val F1-Score: {self.calcF1Score(self.calcAverage(val_precision_history), self.calcAverage(val_recall_history))}"  
            )
        
        with open(self.log_file, 'a') as file:
            file.write("\n")
            file.write(self.getCurrTime())
            file.write("\n")
            file.write(f"Time taken: {self.calcTimeElapsed(start_time)}")
            file.write("\n")
            file.write(avgMsg)
            file.write("\n")
        # print(log_message, end='')
        
    def calcF1Score(self, precision, recall):
        return round((precision * recall * 2) / (precision + recall), 4)
    def calcAverage(self, history):
        return round(sum(history) / len(history), 4)
