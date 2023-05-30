import os
from argparse import ArgumentParser
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

from StandardStrategies.CtoY import CtoY
from StandardStrategies.XtoCtoY import XtoCtoY
from StandardStrategies.XtoY import XtoY
from StandardStrategies.XtoC import XtoC
from continualStrategies.train import train_and_test_continual
from dataset.dataloader import get_loaders
from dataset.dataloader import load_cub200
from explainers.Explainer import Label_Explainer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = join(ROOT_DIR, "results")


def main():
    # parse parameters
    parser = ArgumentParser(description='experiment', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, default="cub200",
                        help='dataset name.', choices=["cub200"])
    parser.add_argument('--classifier', type=str, default="linear",
                        help='classifier name.', choices=["linear", "cnn", "..."])
    parser.add_argument('--featureExtractor', type=str, default="linear",
                        help='classifier name.', choices=["VAE", "CNN", "linear", "resnet"])
    parser.add_argument('--experiment', type=str, default="XtoCtoY",
                        help='classifier name.', choices=["XtoCtoY", "XtoC", "CtoY", "XtoY"])
    parser.add_argument('--continual', action="store_true", default=False,
                        help='standard or continual experiment', )
    parser.add_argument('--batch_size', default=20,
                        help='batchsize', )
    parser.add_argument('--optimizer_name', default="SGD",
                        choices=["SGD", "ADAM"])
    parser.add_argument('-scheduler_step', type=int, default=20,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('--epochs_per_task', default=5,
                        help='epochs_per_task in continual environment', )

    parser.add_argument('-weight_decay', type=float,
                        default=0.0004, help='weight decay for optimizer')
    parser.add_argument('-lr', type=float, default=0.01, help="learning rate")

    args = parser.parse_known_args()[0]

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
    else: torch.manual_seed(seed)

    np.random.seed(seed)

    # get datasets
    train, val, test = load_cub200(join(ROOT_DIR, "data"))

    if args.continual:
        train_and_test_continual(
            train, val, test, device, args, CHECKPOINT_PATH)
    else:

        trainloader, valloader, testloader = get_loaders(
            train, val, test, args.batch_size)
        if args.experiment == "CtoY":
            strategy = CtoY(optimizer_name=args.optimizer_name, model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                            optimizer_hparams={"lr": args.lr, "weight_decay": args.weight_decay, "scheduler_step": args.scheduler_step}, continual=False)
        elif args.experiment == "XtoY":
            strategy = XtoY(optimizer_name=args.optimizer_name, model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                            optimizer_hparams={"lr": args.lr, "weight_decay": args.weight_decay, "scheduler_step": args.scheduler_step}, continual=False)
        elif args.experiment == "XtoCtoY":

            strategy = XtoCtoY(optimizer_name=args.optimizer_name, model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                               optimizer_hparams={"lr": args.lr, "weight_decay": args.weight_decay, "scheduler_step": args.scheduler_step},
                               continual=False,device=device.type)
        elif args.experiment == "XtoC":
            strategy = XtoC(optimizer_name=args.optimizer_name, model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                            optimizer_hparams={"lr": args.lr, "weight_decay": args.weight_decay, "scheduler_step": args.scheduler_step}, continual=False,device=device.type)

        else:
            raise "experiment not defined"
        checkpoint = ModelCheckpoint(save_weights_only=True, mode="max")
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, strategy.name, "standard", args.dataset),
                             # Where to save models
                             # We run on a single GPU (if possible)
                             gpus=1 if str(device) == "cuda" else 0,

                             max_epochs=800,  # How many epochs to train for if no patience is set

                             callbacks=[checkpoint,
                                        # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                        LearningRateMonitor("epoch"),
                                        TQDMProgressBar(1)]  # Log learning rate every epoch
                             )
        #TEST for the metrics

#        print('Before passing to verifier')
#        print(strategy.model.model2.weight.size())
        #explainer=Label_Explainer(strategy, 'data/epoch=188-step=45170.ckpt', testloader) #TODO: fix immediately on cpu
        #explainer.fit(method='GRADCAM')

        trainer.fit(strategy, train_dataloaders=trainloader,
                    val_dataloaders=valloader)


        #explainer=Label_Explainer(strategy, checkpoint.best_model_path, testloader, )
        #explainer.run()


        trainer.test(strategy, dataloaders=testloader)

        #explainer = Label_Explainer(checkpoint.best_model_path, testloader)
        #explainer.run()


if __name__ == '__main__':
    main()
