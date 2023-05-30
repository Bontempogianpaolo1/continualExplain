
from cProfile import label
from training_strategy import er_get_parser
from metrics.metrics import class_concepts
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
    concept_class = class_concepts(cl_model, test_loader, size=(4,200,112), device='cuda')

    t_classes = []

    print('Reference supervision:', args.perc)

    with open(os.path.join(CHECKPOINT_PATH, 'seen_classes_sup'+str(args.perc)+'.txt'), 'r') as file:
        time = 0
        for line in file:
            seen_classes = [int(k) for k in line[3:-2].split(',')]
            t_classes.append(seen_classes)
            concept_class.seen_classes[time] = torch.tensor(seen_classes, dtype=torch.long)
            time += 1

    if not saved_files:
        ## LOAD MODELS AND SAVE CONCEPT PREDICTIONS
        for experience in range(nr_experiences):
            concept_class.checkpoints.append(os.path.join(
                CHECKPOINT_PATH, "model_exp_"+str(experience)+"_perc_"+str(args.perc)+".pt"))

            concept_class.which_classes(torch.tensor(
                t_classes[experience]), experience)

        # UPDATE CONCEPTS for EACH EXPERIENCE and SAVE RESULTS

        sup_dir = CHECKPOINT_PATH+'/perc_'+str(args.perc)
        if not os.path.isdir(sup_dir):
            os.mkdir(sup_dir)
        for t0, t0_classes in enumerate(concept_class.seen_classes):
            # update the concept representation for each classes at starting experience t0

            for time in range(t0, nr_experiences):
                concept_class.update_at_time(
                    time, sup_dir, seen_classes=t0_classes,learning_experience=t0)
    else:
        ## LOAD SAVED FILES IN CONCEPTS
        for t0 in range(nr_experiences):
            for t in range(t0, nr_experiences):
                spath = os.path.join(CHECKPOINT_PATH,
                        'averages_c_class_learnt_at_%i_seen_at_experience_%i_sup'+str(args.perc)+'.pt'%(t0,t))
                classes = concept_class.seen_classes[t0]
                concept_class.c[t, classes]  += torch.load(spath)[t, classes]
    ### BEGIN TESTING KLD ###

    sup_dir='data/perc_'+str(args.perc)

    kl_concepts_at_all_times = overall_kld(concept_class, nr_experiences,dir=sup_dir)

    kl_concepts_at_all_times_0, c = overall_kld(concept_class, nr_experiences, which_class=10,dir=sup_dir)

    _ = overall_kld_real(concept_class, nr_experiences,dir=sup_dir)

    for t in range(3):
        class_specific_kld(concept_class, nr_experiences, t, perc=args.perc,dir=sup_dir)
        singular_kld(concept_class, nr_experiences, t, perc=args.perc,dir=sup_dir)
        #overall_kld(concept_class, nr_experiences, perc=args.perc,dir=sup_dir)
        worst_attributes(kl_concepts_at_all_times[t], nr_experiences, t, perc=args.perc,dir=sup_dir) # over all classes
        worst_attributes(kl_concepts_at_all_times_0[t], nr_experiences, t, which_class=c[t], perc=args.perc,dir=sup_dir) # over only which=0

        real_singular_kld(concept_class, nr_experiences, t, perc=args.perc,dir=sup_dir)

    print('Concluded test')

if __name__ == '__main__':
    main()
