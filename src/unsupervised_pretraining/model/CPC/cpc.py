"""Model based on "Representation Learning with Contrastive Predictive Coding" paper.
   https://arxiv.org/pdf/1807.03748.pdf
"""
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchmetrics import Accuracy

from unsupervised_pretraining.model.CPC import CPCEncoder


class CPCModel(pl.LightningModule):

    def __init__(self, learning_rate, embed_dim, T=49, k=5, inference=False):
        """
        Args:
            learning_rate: Adam optimizer learning rate.
            embed_dim: embedding dimension (default is 512).
            T: overall number of patches from image.
            k: steps for prediction.
        """
        super(CPCModel, self).__init__()
        self.encoder = CPCEncoder()
        self.autoregressive = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(512, embed_dim) for _ in range(k)])
        self.model = nn.ModuleList(list([self.encoder, self.autoregressive, self.Wk]))
        if inference:
            self.pooling = nn.AvgPool2d(49)

        self.emb_dim = embed_dim
        self.num_negative_samples = 5  # magic constant
        self.T = T
        self.k = k
        self.learning_rate = learning_rate
        self.metric = Accuracy()

    def forward(self, X):
        """
        Forward pass of CPC model.
        Takes batch of 3-channel images and passes through encoder part and average pool layer.
        Args:
            X - batch of images with shape [batch_size, 3, 256, 256].
        Returns:
            embedding - mean embeddings for each image with shape [batch_size, emb_dim].
        """
        patches_embeddings = self.encoder(X)
        embeddings = self.pooling(patches_embeddings).squeeze(-1)
        return embeddings

    @staticmethod
    def function(r, z):
        """Function to calculate scores (density ratio).
        Args:
            r: predicted by log-bilinear model with shape [batch_size, self.T - self.k, self.k, emb_dim]
            z: embeddings from encoder with shape [batch_size, self.T - self.k, emb_dim]
        Returns:
            scores: shape [batch_size, self.T - self.k, self.k]
        """
        # do tile
        tiled_z = torch.empty(*r.shape)
        rolled_z = z.clone()
        for idx in range(z.shape[1]):
            rolled_z = torch.roll(rolled_z, 1, 1)
            tiled_z[:, idx, :, :] = z[:, 0:5, :]
        scores = torch.sum(torch.mul(r, tiled_z), dim=-1)
        return scores

    def step(self, X):
        """General step for train/val/test CPC model step.
           Args:
               X - batch of images with shape [batch_size, 3, 256, 256].
           Returns:
               loss - infoNCE loss for current batch.
        """
        batch_size = X.shape[0]
        z = self.encoder(X)  # returns encoded patches with shape [batch_size, self.emb_dim, self.T]

        z_t, z_t_k = z[:, :, :-self.k], z[:, :, -self.k:]
        z_t = z_t.permute(0, 2, 1).to(self.device)
        context_t, _ = self.autoregressive(z_t)
        r = torch.zeros((batch_size, self.T - self.k, self.k, self.emb_dim))
        for idx, linear in enumerate(self.Wk):
            r[:, :, idx, :] = linear(context_t)
        positives = self.function(r, z_t)
        negatives = torch.zeros((batch_size, self.num_negative_samples, self.T - self.k, self.k))
        for idx in range(self.num_negative_samples):
            z_rolled = torch.roll(z_t, 1, dims=0)
            negatives[:, idx, :, :] = self.function(r, z_rolled)
        denominator = torch.cat([negatives, positives.unsqueeze(1)], dim=1)  # [batch_size, self.T - self.k, self.k]
        loss = - torch.mean(positives - torch.logsumexp(denominator, dim=1, keepdim=True))  # InfoNCE Loss
        return loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.step(X)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.step(X)
        self.log("val_loss", loss)
        return loss

    def testing_step(self, batch):
        X, y = batch
        loss = self.step(X)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
