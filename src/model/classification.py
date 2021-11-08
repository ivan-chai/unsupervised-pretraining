import torch.jit
import torch.nn as nn
import torchvision


class ClassificationModel(nn.Module):

    def __init__(self, embed_dim=512, num_classes: int = 10):
        super().__init__()
        # TODO: заменить resnet18 на эмбеддер. В текущем примере dim эмбеддинга равно 256
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, embed_dim)
        jit_model = torch.jit.script(model)
        #  self.embedder = torch.jit.load(path_to_model)
        self.embedder = jit_model
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, X):
        embedding = self.embedder(X)
        out = self.fc(embedding)

        return out
