import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

import torchvision
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

# Define Data Preprocessing
# means and standard deviations ImageNet because the network is pretrained
means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Define transforms to apply to each image
#torch.manual_seed(10)
transf_train = transforms.Compose([ transforms.Resize(224),      # Resizes short size of the PIL image to 256
                              transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                              #transforms.RandomHorizontalFlip(p=0.5),  #1.0
                              transforms.ColorJitter(brightness=0.25, hue=0.25),
                              #transforms.RandomRotation(degrees=(-20,20)),
                              transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])

transf_val = transforms.Compose([ transforms.Resize(224),      # Resizes short size of the PIL image to 256
                              transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])

def get_loaders(data_dir, dataset_name, domain, batch_size):

    if dataset_name == 'pacs':
        # Prepare Dataset
        # Define datasets root
        print('Domain Chosen : {}'.format(domain))
        dir_train = os.path.join(data_dir, 'pacs_official_split', 'train', domain)
        dir_val = os.path.join(data_dir, 'pacs_official_split', 'val', domain)
        dir_test = os.path.join(data_dir, 'pacs_official_split', 'test', domain)

        # Prepare Pytorch train/test Datasets
        train_dataset = torchvision.datasets.ImageFolder(dir_train, transform=transf_val)
        val_dataset = torchvision.datasets.ImageFolder(dir_val, transform=transf_val)
        test_dataset = torchvision.datasets.ImageFolder(dir_test, transform=transf_val)

        # Check dataset sizes
        print(f"No. of Training Examples: {len(train_dataset)}")
        print(f"No. of Validation Examples: {len(val_dataset)}")
        print(f"No. of Test Examples: {len(test_dataset)}")

        # Prepare Dataloaders

        # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

        return train_loader, val_loader, test_loader


def get_loaders_custom(data_dir, dataset_name, domain, mode, transf, shuffle, batch_size, drop_last):

    if dataset_name == 'pacs':
        # Prepare Dataset
        # Define datasets root
        print('Domain Chosen : {}'.format(domain))
        dir = os.path.join(data_dir, 'pacs_official_split', mode, domain)

        # Prepare Pytorch train/test Datasets
        dataset = torchvision.datasets.ImageFolder(dir, transform=transf)

        # Check dataset sizes
        print('No.of {} Examples = {}'.format(mode, len(dataset)))

        # Prepare Dataloaders

        # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)

        return loader


# Data exploration

#print(train_dataset.imgs) # same of print(train_dataset.samples)
# [('Homework3-PACS/PACS/photo/dog/056_0001.jpg', 0),
#  ('Homework3-PACS/PACS/photo/dog/056_0002.jpg', 0) ... ]

#print(train_dataset.classes)
# 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'

#print(train_dataset.class_to_idx)
# {'dog': 0,
#  'elephant': 1,
#  'giraffe': 2,
#  'guitar': 3,
#  'horse': 4,
#  'house': 5,
#  'person': 6}

# dimension of an image 3x227x227
# torch.Size([3, 227, 227])

# plot images distribution
#plotImageDistribution(photo_dataset.targets, art_dataset.targets, cartoon_dataset.targets, sketch_dataset.targets, DATASETS_NAMES, CLASSES_NAMES, show=SHOW_IMG)
