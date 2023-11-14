from utils.arguments import parse_arguments
import os

NUM_FOLDS = 1
BATCH_SIZE = 512
SEED = 42
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
RESAMPLE_RESOLUTION = (224, 224)
BASE_MODEL_PATH = "./models/base_model.pth"

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

args = parse_arguments()
# Access the parsed arguments
MODEL = args.model
DATASET = args.dataset
PRETRAINED_BOOL = args.pretrained
NUM_EPOCHS = args.num_epochs
EARLY_STOPPING_BOOL = args.early_stopping
DATA_AUGMENT_BOOL = args.data_augment
data_folder = os.path.join('/home/emok/sq58_scratch/emok/Data/', DATASET)
print(f"MODEL: {MODEL}\t"
    f"DATASET: {DATASET}\t"
    f"DATA AUGMENT: {DATA_AUGMENT_BOOL}"
)