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
        self.encoder = nn.Sequential(*list(resnet18().children())[:-2])
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
        for b in range(batch_size):
            for i in range(7):
                for j in range(7):
                    patch_emb = self.encoder(X[b, :, 32 * i: 64 + 32 * i, 32 * j: 64 + 32 * j].unsqueeze(0)).squeeze(-1).squeeze(-1)
                    embedding[b, :, (i + j)] = patch_emb
        return embedding

    def to_jit(self):
        """Compile encoder and saves compiled model to predefined path."""
        jit_encoder = jit.script(self.encoder)
        jit.save(jit_encoder, self.encoder_path)

    def _prepare_health_embedding(self):
        pass
        #  TODO: make function for health part of dataset
