import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy


class ResNet18(pl.LightningModule):

    def __init__(self, num_classes, learning_rate, pretrained):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(1000, num_classes)
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, X):
        X = self.model(X)
        X = self.fc(X)
        return F.softmax(X, dim=1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.loss(logits, y)
        return loss

    def validation(self, batch):
        X, y = batch
        probs = self.model(X)
        loss = self.loss(probs, y)
        preds = torch.argmax(probs, dim=1)
        acc = self.metric(preds, y)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.validation(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.validation(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
