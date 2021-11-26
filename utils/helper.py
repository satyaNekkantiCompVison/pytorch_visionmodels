## Helper functions

import torch
import random
import numpy as np
from torchsummary import summary

def get_device():

    cuda = torch.cuda.is_available()
    print("CUDA Available:", cuda)

    device = torch.device("cuda" if cuda else "cpu")

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


def unnormalize(img):
    """
    De-normalize the image.
    """
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    img = img.cpu().numpy().astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * std[i]) + mean[i]

    return np.transpose(img, (1, 2, 0))