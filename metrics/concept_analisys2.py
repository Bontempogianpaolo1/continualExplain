import json
from posixpath import split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
import itertools

from tqdm import tqdm
from dataset.dataloader import load_cub200, get_loaders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Data to be written

# Serializing json
with open('json_data.json', 'r') as outfile:
    data = json.load(outfile)
empty_mask = ['0' for v in range(112)]
#print(list(itertools.combinations(range(112),2)))
for r in range(20):
    print(r)
    for combination in tqdm(list(itertools.combinations(range(112), r))):
        full_mask = empty_mask.copy()
        for bit in combination:
            full_mask[bit] = '1'

        for set in data.keys():
            #print(set)
            elements = set.split(" ")

            if (elements == full_mask):
                print("true")
                print(combination)
                print(data[set])
