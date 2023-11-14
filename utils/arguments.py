import argparse

def parse_arguments():
    # CLI Parser
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run the Mammogram Classifier")

    # Add arguments
    parser.add_argument('--model', type=str, required=True, help='Model Name')
    parser.add_argument('--pretrained', type=bool, default=True, required=False, help='Pretrained Boolean (Default: True)')
    parser.add_argument('--dataset', type=str, required=True, help='Datasets: CBIS-DDSM / CMMD / RSNA / USF / VinDr')
    parser.add_argument('--num_epochs', type=int, default=200, required=False, help='Number of Epochs (Default: 200)')
    parser.add_argument('--early_stopping', action='store_true', required=False, help='Stop training if validation loss doesnt decrease')
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false')
    parser.add_argument('--data_augment', action='store_true', help='Refer to data_loading/data_loader.py for list of augmentation')
    parser.add_argument('--no-data_augment', dest='data_augment', action='store_false')
    parser.set_defaults(data_augment=False)

    return parser.parse_args()