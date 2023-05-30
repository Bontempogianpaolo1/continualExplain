import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from nets.model import YandAttributes


class XtoC(pl.LightningModule):

    def __init__(self, model_hparams, optimizer_name, optimizer_hparams, continual=False, device='cuda'):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = YandAttributes(200,312,515,False, device=device)
        self.name = "XtoC"
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.continual = continual

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters(
            )), lr=self.hparams.optimizer_hparams["lr"], momentum=0.9, weight_decay=self.hparams.optimizer_hparams["weight_decay"])
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.optimizer_hparams["scheduler_step"], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        if self.continual:
            imgs, label, attributes, _ = batch
        else:
            imgs, label, attributes = batch
        attributes = torch.stack(attributes, dim=1).float()
        pred_attributes = self.model(imgs)
        tot_loss = 0
        for i in range(312):
            loss = self.loss_module(pred_attributes[:, i].squeeze().to(dtype=torch.float, device=device), #  type(torch.cuda.FloatTensor),
                                    attributes[:, i].to(dtype=torch.long, device=device)) #type(torch.LongTensor).cuda())
            self.log("train_loss_concept" + str(i), loss,on_epoch=True)
            tot_loss += loss
        self.log("train_loss", tot_loss)
        return tot_loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, label, attributes = batch
        attributes = torch.stack(attributes, dim=1).float()
        pred_attributes = self.model(imgs)
        loss=0
        tot_loss=0
        for i in range(312):
            loss = self.loss_module(pred_attributes[:, i].squeeze().to(dtype=torch.float, device=device),# type(torch.cuda.FloatTensor),
                                    attributes[:, i].to(dtype=torch.long, device=device))  # type(torch.LongTensor).cuda())
            self.log("val_loss_concept"+str(i),loss)
            tot_loss+=loss
        # By default logs it per epoch (weighted average over batches)

        self.log('val_tot_loss', tot_loss)

    def test_step(self, batch, batch_idx):
        imgs, label, attributes = batch
        attributes = torch.stack(attributes, dim=1).float()
        preds = self.model(imgs)
        loss = self.loss_module(preds, attributes).sum()
        self.log('test_loss', loss)
