import os

NUM_FOLDS = 1
BATCH_SIZE = 16
SEED = 42
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
PATIENCE = 15
WEIGHT_DECAY = 1e-2
RESAMPLE_RESOLUTION = (224, 224)
BASE_MODEL_PATH = "./models/base_model.pth"
PRETRAINED_ROI_MODEL_PATH = "./models/pretrained_CBISMassROI.pth"

# FL
NUM_ROUNDS = 60
TRAIN_EPOCHS = 3
DATA_AUGMENT_BOOL = True
MODEL = "resnet50"
PRETRAINED_BOOL = True
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"
PRETRAINED_MODEL_PATH = None # <- Change the starting FL model weights from here

# Use default train/test split
# Train/validation Ratio: 80/20
TRAIN_RATIO = 0.8  # 80% for training
VAL_RATIO = 0.2    # 20% for validation

all_images = []
all_labels = []
train_images = []
val_images = []
train_labels = []
val_labels = []
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []

split_train_accuracy_history = []
split_train_loss_history = []
split_val_accuracy_history = []
split_val_loss_history = []
split_val_precision_history = []
split_val_recall_history = []
split_val_f1_history = []
