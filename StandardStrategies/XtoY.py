import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

from nets.model import base, baseTransformer


class XtoY(pl.LightningModule):

    def __init__(self, model_hparams, optimizer_name, optimizer_hparams, continual=False):
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
        self.model = base(200)
        self.name = "XtoY"
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.continual= continual

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

        preds = self.model(imgs)
        loss = self.loss_module(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, label, attribute = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        imgs, label, attributes = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('test_acc', acc)
        self.log('test_loss', loss)
