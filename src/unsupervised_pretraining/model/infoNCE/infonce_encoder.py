import torch
import torch.nn as nn
import torch.jit as jit
from torchvision.models import resnet18


class InfoNCEEncoder(nn.Module):
    """ResNet-18 encoder for InfoNCE model.
       Dimension of output embedding is (512, 7, 7).
    """
    def __init__(self, encoder_path):
        super(InfoNCEEncoder, self).__init__()
        layers = list(resnet18().children())[1:-2]
        layers.insert(0, nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))  # for gray-scale
        self.encoder = nn.Sequential(*layers)
        self.encoder.add_module("mean_pool", nn.AvgPool2d(2))
        self.encoder_path = encoder_path

    def forward(self, X):
        """Forward pass of encoder.
           Encoder splits grayscale image (256x256) into 7x7 patches with shape 64x64.
           Then each patch passes through encoder.
        """
        # X.shape = [batch_size, channels, height, width]
        batch_size = X.shape[0]
        embedding = torch.zeros((batch_size, 512, 49))
        X = X[:, 0, :, :]
        X = X.unfold(1, 64, 32).unfold(2, 64, 32).reshape(batch_size, 1, 49, 64, 64)  # crops for current batch
        for i in range(49):
            embedding[:, :, i] = self.encoder(X[:, :, i, :, :]).squeeze(-1).squeeze(-1)
        return embedding

    def to_jit(self):
        """Compile encoder and saves compiled model to predefined path."""
        jit_encoder = jit.script(self.encoder)
        jit.save(jit_encoder, self.encoder_path)

    def _prepare_health_embedding(self):
        pass
        #  TODO: make function for health part of dataset
