"""Model based on "Representation Learning with Contrastive Predictive Coding" paper.
   https://arxiv.org/pdf/1807.03748.pdf
"""
import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from torchmetrics import Accuracy

from unsupervised_pretraining.model.infoNCE import InfoNCEEncoder


class CPCTrainModel(pl.LightningModule):

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
        super(CPCTrainModel, self).__init__()
        self.encoder = InfoNCEEncoder(weights_path)
        self.autoregressive = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(512, embed_dim) for _ in range(k)])
        self.model = nn.ModuleList(list([self.encoder, self.autoregressive, self.Wk]))

        self.emb_dim = embed_dim
        self.weights_path = weights_path
        self.num_negative_samples = 5
        self.T = T
        self.k = k
        self.lr = learning_rate
        self.metric = Accuracy()

    def forward(self, X):
        pass

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

    def training_step(self, batch, batch_idx):
        X, y = batch
        batch_size = X.shape[0]
        z = self.encoder(X)  # returns encoded patches with shape [batch_size, self.emb_dim, self.T]

        z_t, z_t_k = z[:, :, :-self.k], z[:, :, -self.k:]
        z_t = z_t.permute(0, 2, 1).to(self.device)
        context_t, _ = self.autoregressive(z_t)
        r = torch.zeros((batch_size, self.T - self.k, self.k, self.emb_dim))
        for idx, linear in enumerate(self.Wk):
            r[:, :, idx, :] = linear(context_t)
        positives = self.function(r, z_t)
        # take 10 negative samples. 10 - is a magic constant
        negatives = torch.zeros((batch_size, self.num_negative_samples, self.T - self.k, self.k))
        for idx in range(self.num_negative_samples):
            z_rolled = torch.roll(z_t, 1, dims=0)
            negatives[:, idx, :, :] = self.function(r, z_rolled)
        denominator = torch.cat([negatives, positives.unsqueeze(1)], dim=1)  # [batch_size, self.T - self.k, self.k]

        loss = - torch.mean(positives - torch.logsumexp(denominator, dim=1, keepdim=True))  # InfoNCE Loss
        self.log("val_loss", loss)
        return loss

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
        positives = self.function(r, z_t)
        # take 10 negative samples. 10 - is a magic constant
        negatives = torch.zeros((batch_size, self.num_negative_samples, self.T - self.k, self.k))
        for idx in range(self.num_negative_samples):
            z_rolled = torch.roll(z_t, 1, dims=0)
            negatives[:, idx, :, :] = self.function(r, z_rolled)
        denominator = torch.cat([negatives, positives.unsqueeze(1)], dim=1)  # [batch_size, self.T - self.k, self.k]

        loss = - torch.mean(positives - torch.logsumexp(denominator, dim=1, keepdim=True))  # InfoNCE Loss
        self.log("val_loss", loss)
        return loss

    def testing_step(self, batch):
        X, y = batch
        batch_size = X.shape[0]
        z = self.encoder(X)  # returns encoded patches with shape [batch_size, self.emb_dim, self.T]

        z_t, z_t_k = z[:, :, :-self.k], z[:, :, -self.k:]
        z_t = z_t.permute(0, 2, 1).to(self.device)
        context_t, _ = self.autoregressive(z_t)
        r = torch.zeros((batch_size, self.T - self.k, self.k, self.emb_dim))
        for idx, linear in enumerate(self.Wk):
            r[:, :, idx, :] = linear(context_t)
        positives = self.function(r, z_t)
        # take 5 negative samples. 5 - is a magic constant
        negatives = torch.zeros((batch_size, self.num_negative_samples, self.T - self.k, self.k))
        for idx in range(self.num_negative_samples):
            z_rolled = torch.roll(z_t, 1, dims=0)
            negatives[:, idx, :, :] = self.function(r, z_rolled)
        denominator = torch.cat([negatives, positives.unsqueeze(1)], dim=1)  # [batch_size, self.T - self.k, self.k]

        loss = - torch.mean(positives - torch.logsumexp(denominator, dim=1, keepdim=True))  # InfoNCE Loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
