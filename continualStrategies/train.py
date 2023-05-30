import torch
import torch.optim as optim
from avalanche.benchmarks.generators import nc_benchmark
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from nets.model import base
from StandardStrategies.XtoC import XtoC
from StandardStrategies.XtoY import XtoY
from StandardStrategies.CtoY import CtoY
from StandardStrategies.XtoCtoY import XtoCtoY
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from dataset.dataloader import get_loaders
import os
from explainers.Explainer import Label_Explainer



def train_and_test_continual(train, val, test, device,args,CHECKPOINT_PATH):
    x = nc_benchmark(train_dataset=train, test_dataset=test, n_experiences=10, task_labels=False)
    stream = x.train_stream
    _, val_loader,test_loader= get_loaders(train,val,test,args.batch_size)
    # strategy= Naive(nn.Linear(),)
    if args.experiment == "CtoY":
        strategy = CtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif args.experiment == "XtoY":
        strategy = XtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif args.experiment == "XtoCtoY":
        strategy = XtoCtoY(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                           optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    elif args.experiment == "XtoC":
        strategy = XtoC(optimizer_name="Adam", model_hparams={"num_classes": 200, "act_fn_name": "relu"},
                        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, continual=True)
    checkpoint = ModelCheckpoint(save_weights_only=True, mode="max")
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, strategy.name, "continual", args.dataset),
                         # Where to save models
                         gpus=1 if str(device) == "cuda" else 0,  # We run on a single GPU (if possible)
                         max_epochs=args.epochs_per_task,  # How many epochs to train for if no patience is set
                         callbacks=[checkpoint,
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch"),
                                    TQDMProgressBar(1)]  # Log learning rate every epoch
                         )
    for experience in stream:
        print('Started experience',experience.task_label)
        t = experience.task_label
        exp_id = experience.current_experience
        training_dataset = experience.dataset
        trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
        trainer.fit(strategy, train_dataloaders=trainloader, val_dataloaders=val_loader)

        explainer = Label_Explainer(checkpoint.best_model_path, test_loader)
        explainer.run()

        trainer.test(strategy, dataloaders=test_loader)
        print('Task {} batch {} -> train'.format(t, exp_id))
        print('This batch contains', len(training_dataset), 'patterns')


def train_and_test(args, train, val, test,device):
    dataloader = DataLoader(train, batch_size=5, shuffle=True, drop_last=True)
    model = base().to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=0.01)
    epochs=20
    for epoch in range(epochs):
        tot_loss=0
        steps=0
        for img, y, attr in tqdm(dataloader):
            #print(y)
            img=img.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            #print(loss)
            steps= steps+1
            tot_loss=tot_loss+loss.item()
            if steps % 50 ==0:
                print(loss)
        print(tot_loss/steps)