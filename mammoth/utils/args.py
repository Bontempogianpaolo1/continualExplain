# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from mammoth.models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str,  default="cub200",
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str,  default="YandConcepts",
                        help='Model name.', choices=['YandConcepts', get_all_models()])

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.0004,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int, default=25,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=420,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true', default=False,
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true', default=True,
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, default=100,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=5,
                        help='The batch size of the memory buffer.')
