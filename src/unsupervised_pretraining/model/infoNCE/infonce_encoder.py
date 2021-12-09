import torch
import torch.nn as nn
import torch.jit as jit
from torchvision.models import resnet18


class InfoNCEEncoder(nn.Module):
    """ResNet-18 encoder for InfoNCE model.
       Dimension of output embedding is (1, 512).
    """
    def __init__(self, encoder_path):
        super(InfoNCEEncoder, self).__init__()
        self.encoder = nn.Sequential(*list(resnet18().children())[:-2])
        self.encoder.model.add_module("mean_pool", nn.AvgPool2d(7))
        self.encoder_path = encoder_path

    def forward(self, X):
        """Forward pass of encoder.
           Encoder splits image (256x256) into 7x7 patches with shape 64x64.
           Then each patch passes through encoder.
        """
        embedding = torch.zeros((7, 7, 512))
        for i in range(7):
            for j in range(7):
                embedding[i, j] = self.encoder(X[32 * i: 64 + 32 * i][32 * j: 62 + 32 * j])
        return embedding

    def _to_jit(self):
        """Compile encoder and saves compiled model to predefined path."""
        jit_encoder = jit.script(self.encoder)
        jit.save(jit_encoder, self.encoder_path)

    def _prepare_health_embedding(self):
        pass
        #  TODO: make function for health part of dataset
