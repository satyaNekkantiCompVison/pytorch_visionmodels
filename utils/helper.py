## Helper functions

import torch
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