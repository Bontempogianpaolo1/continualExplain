import os
import sys
from argparse import ArgumentParser
from os.path import join

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset


#from .training_strategy import train_and_test_continual
#from dataset.dataloader import get_loaders
from dataset.dataloader import load_2MNIST

#from continualStrategies.choose_strategy import choose_model

from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.training import Naive
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.benchmarks.utils import AvalancheDataset



# EM

conf_path = os.getcwd() + "/det_VAEL/"
sys.path.append(conf_path)
sys.path.append(conf_path + '/models')
sys.path.append(conf_path + '/utils')
from det_VAEL.models.vael import MNISTPairsDeepProblogModel
from det_VAEL.models.vael_networks import DetMNISTPairsEncoder, MNISTPairsMLP
from det_VAEL.utils.mnist_utils.mnist_task_VAEL import build_model_dict, build_worlds_queries_matrix
from det_VAEL.utils.graph_semiring import GraphSemiring




ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = join(ROOT_DIR, "results")



def det_loss_function(query_prob, labels, query=True,  
                    query_w=1,
                    sup_w=1, sup=False):

    add_prob = query_prob
   
    # Cross Entropy on the query
    if query:
        target = torch.ones_like(add_prob)
        query_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(add_prob), torch.flatten(target))
    else:
        query_cross_entropy = torch.zeros(size=())

    # Cross Entropy digits supervision
    if sup:
        idxs = labels[labels[:, -1] >= -1][:, -1]  # Index(es) of the labelled image in the current batch
        digit1, digit2 = labels[idxs[0]][:2]  # Correct digit in position 1 and 2, each batch has the same images

        #pred_digit1 = model.facts_probs[idxs, 0, digit1]
        #pred_digit2 = model.facts_probs[idxs, 1, digit2]
        #pred = torch.cat([pred_digit1, pred_digit2])
        #target = torch.ones_like(pred, dtype=torch.float32)
        #label_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(pred), torch.flatten(target))
    else: 
        label_cross_entropy = torch.zeros(size=())

    # Total loss
    loss = query_w * query_cross_entropy + sup_w * label_cross_entropy

    return loss



def main(**kwargs):
    # parse parameters
    parser = ArgumentParser(description='experiment', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, default="cub200", help='dataset name.', choices=["cub200", "MNIST"])
    parser.add_argument('--classifier', type=str, default="linear", help='classifier name.', choices=["linear", "cnn", "..."])
    parser.add_argument('--featureExtractor', type=str, default="linear", help='classifier name.', choices=["VAE", "CNN", "linear", "resnet"])
    parser.add_argument('--experiment', type=str, default="XtoCtoY", help='classifier name.', choices=["XtoCtoY", "XtoC", "CtoY", "XtoY", "2MNIST"])
    parser.add_argument('--continual', action="store_true", default=False, help='standard or continual experiment', )
    parser.add_argument('--batch_size', default=30, help='batchsize', )
    parser.add_argument('--optimizer_name', default="ADAM", choices=["SGD", "ADAM"])
    parser.add_argument('--scheduler_step', type=int, default=20, help='Number of steps before decaying current learning rate by half')
    parser.add_argument('--epochs_per_task', default=5, help='epochs_per_task in continual environment', )

    parser.add_argument('--weight_decay', type=float, default=0.0004, help='weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    parser.add_argument('--seed', type=int, default=10, help='Set random seed.')


    args = parser.parse_known_args()[0]

    # set reproducible results
    seed = args.seed
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

    # get datasets
    if args.dataset == 'MNIST':
        train, val, test = load_2MNIST(args.batch_size)

        #  Create the partitions:
        train_split = []
        test_split  = []
        print('train loader lenght', len(train))
        
        print(type(train.images),type(train.labels))

        print(np.shape(train.images), np.shape(train.labels))

        train_set = TensorDataset(torch.tensor(train.images, dtype=torch.float), torch.tensor(train.labels, dtype= torch.int))

        test.images, test.labels





        for i in range(19):
            mask = (train.labels[:,2] == i)
            train_images, train_labels = torch.tensor(train.images[mask], dtype=torch.float), \
                                         torch.tensor(train.labels[mask], dtype=torch.int)
            exp_labels = train_labels.unsqueeze(2).unsqueeze(3)
            train_images = torch.cat((train_images.view(-1,1,56,28), exp_labels.expand(-1,3, 56,28)), dim=1)
            train_labels = train_labels[:,2]

            dataset_i = TensorDataset(train_images, train_labels)
            train_split.append(AvalancheDataset(dataset_i, task_labels=i))


            mask = (test.labels[:,2] == i)
            test_images, test_labels = torch.tensor(test.images[mask], dtype=torch.float), \
                                       torch.tensor(test.labels[mask], dtype=torch.int)
            exp_labels = test_labels.unsqueeze(2).unsqueeze(3)
            test_images = torch.cat((test_images.view(-1,1,56,28), exp_labels.expand(-1,3, 56,28)), dim=1)
            test_labels = test_labels[:,2]

            dataset_i = TensorDataset(test_images, test_labels)
            test_split.append(AvalancheDataset(dataset_i, task_labels=i))

    else:
        NotImplementedError('Wrong choice of dataset')

    # model
    sequence_len, n_digits =2, 10 
    model_dict = build_model_dict(sequence_len, n_digits)


    w_q = build_worlds_queries_matrix(sequence_len, n_digits)
    w_q = w_q.to(device)


    encoder = DetMNISTPairsEncoder(hidden_channels=64, latent_dim=18,dropout=0.5)
    mlp     = MNISTPairsMLP(in_features=18, n_facts=n_digits * 2)
    Model   = MNISTPairsDeepProblogModel(encoder=encoder, mlp=mlp,
                                        latent_dims=18,
                                        model_dict=model_dict, w_q=w_q, dropout=0.5, is_train=True,
                                        device=device)

    Model.semiring = GraphSemiring(batch_size=args.batch_size, device=Model.device)

    # CL Benchmark Creation
    # x = nc_benchmark(train_dataset=train, test_dataset=test,
    #                  n_experiences=4, task_labels=True, seed=seed)

    x = dataset_benchmark(train_split, test_split)

    train_stream = x.train_stream
    test_stream = x.test_stream

    # Prepare for training & testing
    optimizer = Adam(Model.parameters(), lr=0.001)
    criterion = det_loss_function

    # LOGGERS
    loggers = []

    # log to Tensorboard
    loggers.append(TensorboardLogger())

    # log to text file
    loggers.append(TextLogger(open('log.txt', 'a')))

    # print to stdout
    loggers.append(InteractiveLogger())

    # W&B logger - comment this if you don't have a W&B account
    #loggers.append(WandBLogger(project_name="avalanche", run_name="test"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        cpu_usage_metrics(experience=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=19, save_image=True,
                                stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
        benchmark=x)


    # Continual learning strategy
    cl_strategy = Naive(
        Model, optimizer, criterion, train_mb_size=args.batch_size, train_epochs=2,
        eval_mb_size=32, device=device, evaluator=eval_plugin)


    # train and test loop over the stream of experiences
    results = []

    for train_exp in train_stream:
        cl_strategy.train(train_exp)
        results.append(cl_strategy.eval(test_stream))    

if __name__ == '__main__':
    main()

