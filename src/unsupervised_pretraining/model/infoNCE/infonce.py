"""Model based on "Representation Learning with Contrastive Predictive Coding" paper.
   https://arxiv.org/pdf/1807.03748.pdf
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from torchmetrics import Accuracy

from unsupervised_pretraining.model.infoNCE import InfoNCEEncoder


class InfoNCEModel(pl.LightningModule):

    def __init__(self, learning_rate, embed_dim, weights_path, T=49, k=5):
        """
        Args:
            learning_rate: Adam optimizer learning rate.
            timestamps: number of timestamps.
            predict_t: number of timestamps to predict context vectors.
            embed_dim: embedding dimension (default is 512).
            weights_path: path to encoder weights.
            T: overall number of patches from image.
            k: steps for prediction.
        """
        super(InfoNCEModel, self).__init__()
        self.encoder = InfoNCEEncoder(weights_path)
        self.autoregressive = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(512, embed_dim) for _ in range(k)])
        self.model = nn.ModuleList(list([self.encoder, self.autoregressive, self.Wk]))
        self.function = nn.LogSoftmax()

        self.emb_dim = embed_dim
        self.weights_path = weights_path
        self.T = T
        self.k = k
        self.lr = learning_rate
        self.metric = Accuracy()

    def forward(self, X):
        pass

    def training_step(self, batch, batch_idx):
        X, y = batch
        batch_size = X.shape[0]
        z = self.encoder(X)  # returns encoded patches with shape [batch_size, self.T, self.emb_dim]

    def on_train_epoch_end(self) -> None:
        self.encoder.to_jit()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        batch_size = X.shape[0]
        z = self.encoder(X)  # returns encoded patches with shape [batch_size, self.emb_dim, self.T]

        z_t, z_t_k = z[:, :, :-self.k], z[:, :, -self.k:]
        z_t = z_t.permute(0, 2, 1).to(self.device)
        context_t, _ = self.autoregressive(z_t)
        r = torch.zeros((batch_size, self.T - self.k, self.k, self.emb_dim))
        for idx, linear in enumerate(self.Wk):
            r[:, :, idx, :] = linear(context_t)
        # TODO: z_t_k from shape [batch_size, emb_size, self.k] to [batch_size, self.k, self.k, emb_size]
        positive_y = z_t_k  # here could be the great tiling to [batch_size, self.T - self.k, k, self.emb_dim]
        positives = torch.exp(torch.mul(r, positive_y))  # [batch_size, self.T - self.k, self.emb_dim]

        # take 10 negative samples. 10 - is a magic constant
        negatives = torch.zeros((batch_size, 10, self.T - self.k, self.k, self.emb_dim))
        target_neg = torch.roll(z, shifts=1, dim=0)
        for idx in range(10):
            negative_y = target_neg  # do magic tile
            negatives[:, idx, :, :, :] = torch.exp(torch.mul(r, negative_y))
        denominator = torch.cat([negatives, positives], dim=1).sum(dim=1)  # [batch_size, self.T - self.k, self.k]

        loss = - torch.mean(torch.log(positives / denominator))  # InfoNCE Loss
        return loss

    def testing_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
