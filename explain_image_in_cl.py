
from cProfile import label
from training_strategy import er_get_parser
from metrics.metrics import class_concepts, average_concepts
from metrics.kld_visualizations import *
import argparse
import random
from training_strategy import Er_CBM
from mammoth.utils.training import *
from mammoth.models.utils.continual_model import ContinualModel
from mammoth.utils.args import *
from mammoth.utils.buffer import Buffer
from nets.model import YandAttributes, YandConcepts
from mammoth.utils.loggers import CsvLogger
from mammoth.utils.loggers import *
from mammoth.utils.tb_logger import *
from mammoth.utils.status import progress_bar, create_stash
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
import seaborn as sns
import pandas as pd
from explainers.Explainer import Label_Explainer
import matplotlib.pyplot as plt

from avalanche.benchmarks.generators import nc_benchmark
from tqdm import tqdm
from dataset.dataloader import load_cub200, get_loaders
from settings import DATA_DIR, ROOT_DIR, CHECKPOINT_PATH
conf_path = os.getcwd() + "/mammoth/"
sys.path.append(conf_path)
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/models')
sys.path.append(conf_path + '/utils')


#from mammoth.utils.main import lecun_fix, parse_args
#from mammoth.utils.main import *


def main():
    # parse parameter
    # set reproducible results
    seed = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

    # VERY BAD INITIALIZATION #TODO BETTER!

    parser = er_get_parser()  # USEFULL THINGS FOR ER MODEL

    parser.add_argument('--cbm_model', default='XtoCtoY',
                        help='Insert type of CBM used for train and test')

    parser.add_argument('--perc', type=int, default=10,
                        help="percentage-of-supervised-attributes")

    args = parser.parse_known_args()[0]

    # get datasets
    train, val, test = load_cub200(DATA_DIR, perc_supervised=args.perc)

    #train_and_test_continual(train, val, test, args, CHECKPOINT_PATH, device)
    compute_statistics(train, val, test, args, CHECKPOINT_PATH, 4, device)


def compute_statistics(train, val, test, args, CHECKPOINT_PATH, nr_experiences, device='cuda',
                       saved_files=False):

    backbone = YandConcepts(200, n_attributes=112,
                            device=device, classify=True)
    cl_model = Er_CBM(backbone, nn.CrossEntropyLoss(), args, None).to(device)
    _, _, test_loader = get_loaders(train, val, test, args.batch_size)
    concept_class = class_concepts(
        cl_model, test_loader, size=(4, 200, 112), device='cuda')

    t_classes = []

    print('Reference supervision:', args.perc)

    with open(os.path.join(CHECKPOINT_PATH, 'seen_classes_sup'+str(args.perc)+'.txt'), 'r') as file:
        time = 0
        for line in file:
            seen_classes = [int(k) for k in line[3:-2].split(',')]
            t_classes.append(seen_classes)
            concept_class.seen_classes[time] = torch.tensor(
                seen_classes, dtype=torch.long)
            time += 1
    class0 = concept_class.seen_classes[0][0]
    while True:
        data = next(iter(concept_class.dataloader))
        imgs, lab, attr, _ = data
        labf = lab[lab == class0]
        if labf.shape[0] > 0:
            imgs = imgs[lab == class0]
            image = imgs[0]
            break

    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_0_perc_100.pt", test_loader=test_loader)
    explainer.explain(image, "perc_100_experience0", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_1_perc_100.pt", test_loader=test_loader)
    explainer.explain(image, "perc_100_experience1", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_2_perc_100.pt", test_loader=test_loader)
    explainer.explain(image, "perc_100_experience2", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_3_perc_100.pt", test_loader=test_loader)
    explainer.explain(image, "perc_100_experience3", class0)
    #expla
    #
    # iner.run()

    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_0_perc_0.pt", test_loader=test_loader)
    explainer.explain(image, "perc_0_experience0", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_1_perc_0.pt", test_loader=test_loader)
    explainer.explain(image, "perc_0_experience1", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_2_perc_0.pt", test_loader=test_loader)
    explainer.explain(image, "perc_0_experience2", class0)
    explainer = Label_Explainer(
        cl_model, "/data/emarcona/results/model_exp_3_perc_0.pt", test_loader=test_loader)
    explainer.explain(image, "perc_0_experience3", class0)


if __name__ == '__main__':
    main()
