import torch
from torch.nn import functional as F
import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, backbone, loss, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.loss = loss

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)

        if batch_idx % 10 == 0:
            tag = 'true_wrong'  # tag based on success
            mesh = torch.swapaxes(x[0:1], 0, 1)
            tensorboard = self.logger.experiment
            tensorboard.add_mesh(tag, mesh)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
