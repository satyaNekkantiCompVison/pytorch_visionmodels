import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


import csv
import os
import zipfile
from io import BytesIO
import requests
from PIL import Image
from tqdm import notebook

class AlbumentationImageDataset(Dataset):
  def __init__(self, image_list, train= True,Aug=None,mean=None, std=None):
      self.image_list = image_list
      # self.aug = A.Compose({
      #            A.Rotate (limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
      #            A.Sequential([A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by default
      #            A.RandomCrop(32,32)]),
      #            A.CoarseDropout(1, 16, 16, 1, 16, 16,fill_value=0.473363, mask_fill_value=None),
      #            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      #         })
      self.aug = Aug
      self.norm = A.Compose({A.Normalize(mean, std),
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
  train_loader = DataLoader(AlbumentationImageDataset(trainset, train=True,Aug=AugTransforms,mean=[0.49139968, 0.48215841, 0.44653091],std=[0.24703223, 0.24348513, 0.26158784]),batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

  return train_loader


def get_test_loader(BATCH_SIZE=128):
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False)

  test_loader = DataLoader(AlbumentationImageDataset(testset, train=False,Aug=None,mean=[0.49139968, 0.48215841, 0.44653091],std=[0.24703223, 0.24348513, 0.26158784]), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

  return test_loader



class TinyImageNet(Dataset): # Used the Idea from https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb
    """
    Tiny ImageNet Dataset class.
    """

    def __init__(self, root, train=True, transform=None, download=False, train_split=0.7):
        self.root = root
        self.transform = transform
        self.data_dir = "tiny-imagenet-200"

        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download_and_extract_archive()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.image_paths = []
        self.targets = []

        idx_to_class, class_id = self.get_classes()

        self.classes = list(idx_to_class.values())

        # train images
        train_path = os.path.join(self.root, self.data_dir, "train")
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, "images")
            for image in os.listdir(train_images_path):
                if image.endswith(".JPEG"):
                    self.image_paths.append(os.path.join(train_images_path, image))
                    self.targets.append(class_id[class_dir][0])

        # val images
        val_path = os.path.join(self.root, self.data_dir, "val")
        val_images_path = os.path.join(val_path, "images")
        with open(os.path.join(val_path, "val_annotations.txt")) as val_ann:
            for line in csv.reader(val_ann, delimiter="\t"):
                self.image_paths.append(os.path.join(val_images_path, line[0]))
                self.targets.append(class_id[line[1]][0])

        self.indices = np.arange(len(self.targets))

        random_seed = 42
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        split_idx = int(len(self.indices) * train_split)
        self.indices = self.indices[:split_idx] if train else self.indices[split_idx:]

    def get_classes(self):
        """
        Get class labels mapping
        """
        id_dict = {}
        all_classes = {}
        for i, line in enumerate(open(os.path.join(self.root, "tiny-imagenet-200/wnids.txt"), "r")):
            id_dict[line.replace("\n", "")] = i

        idx_to_class = {}
        class_id = {}
        for i, line in enumerate(open(os.path.join(self.root, "tiny-imagenet-200/words.txt"), "r")):
            n_id, word = line.split("\t")[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            idx_to_class[value] = all_classes[key].replace("\n", "").split(",")[0]
            class_id[key] = (value, all_classes[key])

        return idx_to_class, class_id

    def _check_integrity(self) -> bool:
        """
        Check if Tiny ImageNet data already exists.
        """
        return os.path.exists(os.path.join(self.root, self.data_dir))

    def download_and_extract_archive(self):
        """
        Download and extract Tiny ImageNet data.
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        res = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip", stream=True)
        print("Downloading Tiny ImageNet Data")

        with zipfile.ZipFile(BytesIO(res.content)) as zip_ref:
            for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path=self.root)

    def __getitem__(self, idx):
        image_idx = self.indices[idx]
        filepath = self.image_paths[image_idx]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.targets[image_idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)



def get_train_loader_tinyImageNet(BATCH_SIZE =128,AugTransforms=None):
  trainset = TinyImageNet(root='./data', train=True, download=True)
  train_loader = DataLoader(AlbumentationImageDataset(trainset, train=True,Aug=AugTransforms,mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262]),batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

  return train_loader


def get_test_loader_tinyImageNet(BATCH_SIZE=128):
  testset = TinyImageNet(root='./data', train=False, download=True)

  test_loader = DataLoader(AlbumentationImageDataset(testset, train=False,Aug=None,mean=[0.4802, 0.4481, 0.3975],std=[0.2302, 0.2265, 0.2262]), batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=1)

  return test_loader