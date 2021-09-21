import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import random


def get_tag(t, p):
    pre = '-'
    post = 'right'
    if t == 0:
        pre = '+'
    if p != t:
        post = 'wrong'
    return f'{pre}_{post}'


class Classifier(pl.LightningModule):
    def __init__(self, backbone, loss, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.loss = loss
        self.train_log_interval = 5
        self.val_log_interval = 10

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)

        # logging
        if batch_idx % self.train_log_interval == 0:
            tensorboard = self.logger.experiment
            i = random.randint(0, x.shape[0]-1)  # batch index
            y_hat_i = y_hat[i]
            pred_soft = torch.exp(y_hat_i)
            pred_max = torch.argmax(pred_soft)
            tag = get_tag(y[i], pred_max)
            mesh = torch.swapaxes(x, 1, 2)[i:i+1]

            tensorboard.add_mesh(tag, mesh)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        pass
        # x, y = batch
        # y_hat = self.backbone(x)
        # loss = self.loss(y_hat, y)
        # self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
