# TODO: Universalize health embedding checking
# TODO: bug fix: tensor device mismatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy

from unsupervised_pretraining.utils.disk import load_file
from unsupervised_pretraining.model.CPC.cpc_model import CPCModel


class CPCClassificationModel(pl.LightningModule):

    def __init__(self, model_url, emb_url, num_classes, embed_dim, learning_rate,
                 T, k, inference, weights_path, health_emb_path, health_dataset):
        """ Модель с выходным линейным слоем для задач многоклассовой классификации.

        :param model_url: ссылка на модель.
        :param emb_url: ссылка на проверочный эмбеддинг.
        :param num_classes: количество классов (для STL-10 10 классов).
        :param weights_path: путь к весам претренированной модели.
        :param health_emb_path: путь к проверочному эмбеддингу.
        """
        super(CPCClassificationModel, self).__init__()

        #  self._load_model(model_url, emb_url, weights_path, health_emb_path)
        cpc_model = CPCModel(learning_rate, embed_dim, T, k, inference)
        cpc_model.load_state_dict(torch.load(weights_path))
        self.encoder = cpc_model.encoder
        self.pooling = nn.AvgPool1d(49)
        self.head = nn.Linear(cpc_model.emb_dim, num_classes)
        self.model = nn.Sequential(self.encoder, self.pooling, self.head)
        #  ignore
        #  self.health_emb = torch.load(health_emb_path)
        #  self._check_model_health(health_dataset)
        self.lr = cpc_model.learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    @staticmethod
    def _load_model(model_url, emb_url, weights_save_path, health_emb_save_path):
        model = load_file(model_url)
        emb = load_file(emb_url)
        with open(weights_save_path, "wb") as fout:
            fout.write(model)
        with open(health_emb_save_path, "wb") as fout:
            fout.write(emb)

    def _check_model_health(self, health_dataset):
        input = torch.load(health_dataset)
        model_emb = self.model[0](input)
        if not torch.allclose(model_emb, self.health_emb, atol=1e-5):
            raise Exception("Model is corrupted")

    def forward(self, X):
        X = X[:, 0, :, :].unsqueeze(1)
        X = self.encoder(X)
        logits = self.head(self.pooling(X).squeeze(-1))
        return logits
        #return F.softmax(X, dim=1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.loss(logits, y)
        return loss

    def validation(self, batch):
        X, y = batch
        X = X[:, 0, :, :].unsqueeze(1)
        X = self.encoder(X)
        X = X.to(torch.device("cuda:0"))  # dirty hack. Otherwise, tensor will be on CPU.
        logits = self.head(self.pooling(X).squeeze(-1))
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
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
