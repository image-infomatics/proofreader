import torch
# import pytorch_lightning as pl


def get_tag(t, p):
    pre = 'false'
    post = 'correct'
    if t == 0:
        pre = 'true'
    if p != t:
        post = 'wrong'
    return f'{pre}_{post}'


def predict_class(y_hat):
    pred_soft = torch.exp(y_hat)
    pred_max = torch.argmax(pred_soft, axis=1)
    return pred_max


def get_accuracy(y, pred):
    correct = (y == pred).type(torch.float32)
    total_acc = torch.mean(correct)
    true_acc = torch.mean(correct[y == 1])
    false_acc = torch.mean(correct[y == 0])
    return {'total_acc': total_acc.item(), 'true_acc': true_acc.item(), 'false_acc': false_acc.item()}


# class Classifier(pl.LightningModule):
#     def __init__(self, backbone, loss, learning_rate):
#         super().__init__()
#         self.save_hyperparameters()
#         self.backbone = backbone
#         self.loss = loss

#     def forward(self, x):
#         # use forward for inference/predictions
#         embedding = self.backbone(x)
#         return embedding

#     def training_step(self, batch, batch_idx):
#         # step
#         x, y = batch
#         y_hat = self.backbone(x)
#         loss = self.loss(y_hat, y)
#         self.log('train_loss', loss, on_epoch=True)

#         # accuracies
#         pred = predict_class(y_hat)
#         accs = get_accuracy(y, pred)
#         self.log('train_accuracy', accs['total_acc'], on_epoch=True)

#         # # log meshes
#         # if batch_idx % 10 == 0:
#         #     tb = self.logger.experiment
#         #     i = random.randint(0, x.shape[0]-1)  # batch index
#         #     tag = get_tag(y[i], pred[i])
#         #     mesh = torch.swapaxes(x, 1, 2)[i:i+1]
#         #     tb.add_mesh(tag, mesh)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.backbone(x)
#         loss = self.loss(y_hat, y)
#         self.log('valid_loss', loss, on_epoch=True)

#         # accuracies
#         pred = predict_class(y_hat)
#         accs = get_accuracy(y, pred)
#         self.log('valid_accuracy', accs['total_acc'], on_epoch=True)

#     def test_step(self, batch, batch_idx):
#         pass
#         # x, y = batch
#         # y_hat = self.backbone(x)
#         # loss = self.loss(y_hat, y)
#         # self.log('test_loss', loss)

#     def configure_optimizers(self):
#         # self.hparams available because we called self.save_hyperparameters()
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
