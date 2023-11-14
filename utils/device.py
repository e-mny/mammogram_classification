import torch

def initialize_device():
    """Check and initialize GPU if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"DEVICE: {device}")
    return device
