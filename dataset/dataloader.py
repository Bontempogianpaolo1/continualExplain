import os
from os.path import join

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from dataset.Cub200 import CUBDataset



"""
CUB200 Pytorch Dataset: Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an
extended version of the CUB-200 dataset, with roughly double the number of
images per class and new part location annotations. For detailed information
about the dataset, please check the official website:
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
"""


def load_cub200(path,perc_supervised):
    N_ATTRIBUTES = 112
    N_CLASSES = 200
    data_size = 38
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resol = 299
    print(path)
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(resol),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # implicitly divides by 255
        normalize
    ])

    test_transform = transforms.Compose([
        #transforms.Resize((resized_resol, resized_resol)),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),  # implicitly divides by 255
        normalize
        #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])
    trainset = CUBDataset(join(path, "trainprocessed.pkl"), use_attr=True, no_img=False, uncertain_label=False, image_dir=join(
        path, join("CUB_200_2011", "images")), n_class_attr=N_ATTRIBUTES, transform=train_transform, train=True,perc_supervised=perc_supervised)
    testset = CUBDataset(join(path, "testprocessed.pkl"), use_attr=True, no_img=False, uncertain_label=False, image_dir=join(
        path, join("CUB_200_2011", "images")), n_class_attr=N_ATTRIBUTES, transform=test_transform, train=False,perc_supervised=perc_supervised)
    valset = CUBDataset(join(path, "valprocessed.pkl"), use_attr=True, no_img=False, uncertain_label=False, image_dir=join(
        path, join("CUB_200_2011", "images")), n_class_attr=N_ATTRIBUTES, transform=test_transform, train=False,perc_supervised=perc_supervised)

    return trainset, valset, testset


def get_loaders(trainset, valset, testset, batch_size):
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=batch_size,
                           shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

    return trainloader, valloader, testloader
