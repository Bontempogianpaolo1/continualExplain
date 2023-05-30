import json
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

train, val, test = load_cub200(os.path.join(ROOT_DIR, "data"))

train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
#lst =dict((v,0) for v in itertools.product([0, 1], repeat=2))
#print(lst)
d = {}
concepts_statistics = [0 for i in range(10)]
for i, data in enumerate(train_loader):
    #extract data
    inputs, labels, concepts = data
    concepts = torch.stack(concepts, dim=1)
    try:
        for i in range(100):
            c = str(concepts[i].numpy())
            c=c.replace("[","")
            c=c.replace("]","")
            c=c.replace("\n","")
            if not c in d.keys():
                d[c] = 1
            else:
                d[c] = d[c]+1
        print(d)
    except Exception:
        print("errore")
print(d)


# Data to be written

# Serializing json
json_object = json.dumps(d, indent=4)
print(json_object)
with open('json_data.json', 'w') as outfile:
    outfile.write(json_object)
