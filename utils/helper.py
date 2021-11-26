## Helper functions

import torch
import random
import numpy as np
from torchsummary import summary

def get_device():

    cuda = torch.cuda.is_available()
    print("CUDA Available:", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
        BATCH_SIZE=256
    else:
        BATCH_SIZE=256

    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def model_summary(model, input_size):
    """
    Summary of the model.
    """
    summary(model, input_size=input_size)



def seed_everything(seed: int):
    """
    Seed everything for reproducibility and deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False