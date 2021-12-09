import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchmetrics import Accuracy

from unsupervised_pretraining.model.infoNCE import InfoNCEEncoder


class InfoNCE(pl.LightningModule):

    def __init__(self, learning_rate, timestamps, emb_dim, weights_path):
        super(InfoNCE, self).__init__()

        self.encoder = InfoNCEEncoder(weights_path)
        self.autoregressive = nn.GRU(hidden_size=512)
        self.Wk = nn.ModuleList([nn.Linear(512, emb_dim) for _ in range(timestamps)])

        self.emb_dim = emb_dim
        self.weights_path = weights_path
        self.lr = learning_rate
        self.metric = Accuracy()

    def forward(self, X):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def testing_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
