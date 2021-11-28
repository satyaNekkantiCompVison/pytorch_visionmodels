import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class AlbumentationImageDataset(Dataset):
  def __init__(self, image_list, train= True,Aug=None):
      self.image_list = image_list
      # self.aug = A.Compose({
      #            A.Rotate (limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
      #            A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by default
      #            A.RandomCrop(32,32)]),
      #            A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363, mask_fill_value=None),
      #            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      #         })
      self.aug = Aug
      self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      })
      self.train = train
        
  def __len__(self):
      return (len(self.image_list))

  def __getitem__(self, i):
      
      image, label = self.image_list[i]
      
      if self.train:
        #apply augmentation only for training
        image = self.aug(image=np.array(image))['image']
      else:
        image = self.norm(image=np.array(image))['image']
      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      return torch.tensor(image, dtype=torch.float), label


def get_train_loader(BATCH_SIZE =128,AugTransforms=None):
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=False )
  train_loader = DataLoader(AlbumentationImageDataset(trainset, train=True,Aug=AugTransforms), batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

  return train_loader


def get_test_loader(BATCH_SIZE=128):
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False)

  test_loader = DataLoader(AlbumentationImageDataset(testset, train=False,Aug=None), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

  return test_loader