# TODO: make method for jit model creation
import torch
import torch.nn as nn
from torchvision.models import resnet18


class CPCEncoder(nn.Module):
    """ResNet-18 encoder for CPC model.
       Dimension of output embedding is (512, 7, 7).
    """
    def __init__(self):
        super(CPCEncoder, self).__init__()
        layers = list(resnet18().children())[1:-2]
        layers.insert(0, nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))  # for gray-scale
        self.encoder = nn.Sequential(*layers)
        self.encoder.add_module("mean_pool", nn.AvgPool2d(2))
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def forward(self, X):
        """Forward pass of encoder.
           Encoder splits grayscale image (256x256) into 7x7 patches with shape 64x64.
           Then each patch passes through encoder.

           Args:
               X - batch with input image with shape = [batch_size, 3, 256, 256].
           Returns:
               embedding - images' embedding tensor with shape [batch_size, emb_dim, 49].
        """
        # X.shape = [batch_size, channels, height, width]
        batch_size = X.shape[0]
        embedding = torch.zeros((batch_size, 512, 49))
        X = X[:, 0, :, :]  # take one channel of grayscale image.
        X = X.unfold(1, 64, 32).unfold(2, 64, 32).reshape(batch_size, 1, 49, 64, 64)  # crops for current batch
        for i in range(49):
            embedding[:, :, i] = self.encoder(X[:, :, i, :, :]).squeeze(-1).squeeze(-1)
        return embedding
