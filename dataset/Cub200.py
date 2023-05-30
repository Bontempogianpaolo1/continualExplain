import pickle

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
N_ATTRIBUTES = 112
import torch

class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None,
                 train=True, perc_supervised=100):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.data.extend(pickle.load(open(pkl_file_paths, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.is_train = train
        self.data = np.array(self.data)
        self.targets = [x["class_label"] for x in self.data]
        self.attributes = [x["attribute_label"] for x in self.data]
        self.perc_supervised = perc_supervised
        if self.uncertain_label:
            self.attributes = [x["uncertain_attribute_label"]
                               for x in self.data]
        else:
            self.attributes = [x["attribute_label"] for x in self.data]
        self.data = [x["img_path"] for x in self.data]
        tot_imgs = len(self.targets)
        images_supervised = int(tot_imgs*self.perc_supervised/100)
        print(images_supervised)
        self.flagvector = np.zeros(len(self.data))
        print(self.flagvector.shape)
        self.flagvector[images_supervised:] = 0
        self.flagvector[:images_supervised] = 1
        self.flagvector = self.flagvector[torch.randperm(len(self.data))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_label, attr, is_supervised = self.data[idx], self.targets[
            idx], self.attributes[idx], self.flagvector[idx]
        # Trim unnecessary paths
        #img_path=img_path.replace("/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/", "/homes/gbontempo/ExAIProject/data/")

        img_path = img_path.replace(
            "/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/images", self.image_dir)
        #img_path=img_path.replace("/home/emanuele/Programmi/Machine_Learning/ExAIProject/", "/data/emarcona/")
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, class_label, attr, is_supervised
