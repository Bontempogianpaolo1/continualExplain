import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import os
from settings import DATA_DIR, CHECKPOINT_PATH, ROOT_DIR
from avalanche.benchmarks.generators import nc_benchmark
from tqdm import tqdm
from dataset.dataloader import load_cub200, get_loaders
# ROOT_DIR = '/data/emarcona/data/' # os.path.dirname(os.path.abspath(__file__))
conf_path = os.getcwd() + "/mammoth/"
sys.path.append(conf_path)
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/models')
sys.path.append(conf_path + '/utils')

from mammoth.utils.status import progress_bar, create_stash
from mammoth.utils.tb_logger import *
from mammoth.utils.loggers import *
from mammoth.utils.loggers import CsvLogger
from nets.model import YandAttributes, YandConcepts
from mammoth.utils.buffer import Buffer
from mammoth.utils.args import *
from mammoth.models.utils.continual_model import ContinualModel

from mammoth.utils.training import *
#from mammoth.utils.main import lecun_fix, parse_args
#from mammoth.utils.main import *

from argparse import Namespace
from typing import Tuple

from explainers.Explainer import Label_Explainer

import os
import random
import argparse

from metrics.metrics import class_concepts


#############################################################

def er_get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    return parser


class Er_CBM(ContinualModel):
    NAME = 'er_cbm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er_CBM, self).__init__(backbone, loss, args, transform)
        #self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cbm_model = args.cbm_model
        self.perc = args.perc
        self.loss_concept_module = nn.BCELoss()

    def loss_fn(self, p_concepts, p_labels, concepts, labels, is_supervised):

        tot_loss = 0
        loss_concepts = 0
        loss_prediction = 0
        if p_concepts is not None:
            # Loss on concepts
            p_concepts = p_concepts[is_supervised == 1]
            concepts = concepts[is_supervised == 1]
            if p_concepts.shape[0] > 0:
                loss_concepts += self.loss_concept_module(p_concepts, concepts.to(
                    dtype=torch.float, device=torch.device(self.device)))
        if p_labels is not None:
            # Loss on Labels
            loss_prediction += self.loss(p_labels, labels)
        tot_loss = loss_prediction + loss_concepts
        return tot_loss, loss_prediction, loss_concepts

    def observe(self, inputs, concepts, labels, is_supervised):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        # if not self.buffer.is_empty():
        # buf_inputs, buf_labels = self.buffer.get_data(
        #     self.args.minibatch_size, transform=self.transform)
        # inputs = torch.cat((inputs, buf_inputs))
        # labels = torch.cat((labels, buf_labels))

        p_labels, p_concepts = self.net(inputs)

        tot_loss, loss_pred, loss_concept = self.loss_fn(
            p_concepts, p_labels, concepts, labels, is_supervised)
        tot_loss.backward()

        self.opt.step()
        pred_concepts_discretized = p_concepts
        pred_concepts_discretized[p_concepts > 0.5] = 1
        pred_concepts_discretized[p_concepts < 0.5] = 0
        acc_concepts = torch.sum(
            pred_concepts_discretized == concepts)/(inputs.shape[0]*112)
        y_pred = torch.argmax(p_labels, dim=1)
        acc_labels = torch.sum(y_pred == labels)/(inputs.shape[0])
        # self.buffer.add_data(examples=not_aug_inputs,
        #                    labels=labels[:real_batch_size])
        try:
            print("tot loss: " + str(tot_loss.item())+" prediction loss: " +
                  str(loss_pred.item()) + " concept loss " + str(loss_concept.item()))
        except:
            print("tot loss: " + str(tot_loss.item())+" prediction loss: " +
                  str(loss_pred.item()))
        print("accuracy concept on train " + str(acc_concepts))
        print("accuracy on label " + str(acc_labels))

        print("-"*30)

        return tot_loss.item()


def main():
    # parse parameters

    # TODO:  CREATE A MODULE FOR PARSER

    parser = er_get_parser()  # USEFULL THINGS FOR ER MODEL

    parser.add_argument('--cbm_model', default='XtoCtoY',
                        help='Insert type of CBM used for train and test')

    parser.add_argument('--perc', type=int, default=100,
                        help="percentage-of-supervised-attributes")

    args = parser.parse_known_args()[0]

    # set reproducible results
    seed = 420
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

    # get datasets
    train, val, test = load_cub200(DATA_DIR, args.perc)

    train_and_test_continual(train, val, test, args, CHECKPOINT_PATH, device)
    # TODO: add something here ?
    # TEST for the metrics


def train_and_test_continual(train, val, test, args, CHECKPOINT_PATH, device='cuda'):

    backbone = YandConcepts(200, n_attributes=112,
                            device=device, classify=True)

    cl_model = Er_CBM(backbone, nn.CrossEntropyLoss(), args, None).to(device)
    # insert preprocess on dataset

    x = nc_benchmark(train_dataset=train, test_dataset=test,
                     n_experiences=4, task_labels=True, seed=args.seed)
    stream = x.train_stream
    # print(np.shape(test))
    # print(len(test))
    _, val_loader, test_loader = get_loaders(train, val, test, args.batch_size)

    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger('generic_continual', 'CUB-200', 'CBM')
    if args.tensorboard:
        tb_logger = TensorboardLogger(args,  'generic_continual')

    # select only 10 exemplars from each experience
    #drift_concepts = drifting_concepts(cl_model, test_loader, size=(5,10,112))

    t_classes = []

    print('## STARTING TRAINING ##')
    print('-- CBM model with', args.perc, ' supervision')
    print('-- experiences: 4')
    print('-- seed:', args.seed)

    for experience in stream:
        print('Started experience', experience.task_label)

        # Standard Avalanche initialization
        t = experience.task_label
        exp_id = experience.current_experience

        t_classes.append(experience.classes_in_this_experience)

        training_dataset = experience.dataset
        train_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=args.batch_size, shuffle=True)

        #####  Fit the model in Mammoth  #####
        cl_model.net.train()

        for epoch in range(cl_model.args.n_epochs):
            for i, data in enumerate(train_loader):
                # extract data
                inputs, labels, concepts, is_supervised, task_label = data
                inputs, labels = inputs.to(
                    cl_model.device), labels.to(cl_model.device)
                concepts = torch.stack(concepts, dim=1).to(
                    cl_model.device, dtype=torch.long)

                # compute step
                # try:
                loss = cl_model.observe(
                    inputs, concepts, labels, is_supervised=is_supervised)

                progress_bar(i, len(train_loader), epoch, t, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss=loss, name='total_loss', args=args, epoch=epoch,
                                       task_number=t, iteration=i)
                # except BaseException as err:
                #    print("errore")

            if cl_model.scheduler is not None:
                #cl_model.scheduler.step()
                print("lr "+str(cl_model.scheduler.get_last_lr()))
            # if hasattr(cl_model, 'end_task'):
            #   cl_model.end_task(dataset)
        #accs = evaluate(cl_model, x)
        # results.append(accs[0])
        # results_mask_classes.append(accs[1])

        #mean_acc = np.mean(accs, axis=1)
        #print_mean_accuracy(mean_acc, t + 1, "class_il")

        # if args.csv_log:
        #   csv_logger.log(mean_acc)
        # if args.tensorboard:
            #tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
        # END OF EPOCH
        print('Concluded epoch:', epoch)
        # Insert the savepoint for each experience
        torch.save(cl_model.state_dict(), os.path.join(
            CHECKPOINT_PATH, "model_exp_"+str(t)+"_perc_"+str(args.perc)+".pt"))

        # calculate new concept map at time t
        #avg_concept.update_at_time(cl_model, t)

        #best_model_path = None
        #explainer = Label_Explainer(cl_model, best_model_path, test_loader, debug=True)
        # explainer.fit('GRADCAM')

        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))

    ### SAVE TIME AND SEEN CLASSES ###

    f = open(os.path.join(CHECKPOINT_PATH,
             'seen_classes_sup%i.txt' % args.perc), 'w')
    for i, ele in enumerate(t_classes):
        f.write(str(i)+','+str(ele)+'\n')
    f.close()


if __name__ == '__main__':
    main()
