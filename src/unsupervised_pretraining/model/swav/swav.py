import numpy as np

import torch
from torch import nn
from pytorch_lightning import LightningModule
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torch import distributed as dist

from argparse import ArgumentParser

from swav_resnet import resnet18

class SwAV(LightningModule):
    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        hidden_mlp: int = 2048,
        feat_dim: int = 512,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        nmb_prototypes: int = 3000,
        freeze_prototypes_epochs: int = 1,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        crops_for_assign: tuple = (0, 1),
        nmb_crops: tuple = (2, 6),
        first_conv: bool = True,
        maxpool1: bool = True,
        exclude_bn_bias: bool = False,
        start_lr: float = 3e-4,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        epsilon: float = 0.05,
        **kwargs
    ):
        """
        Args:
            num_samples: number of image samples used for training
            batch_size: batch size
            dataset: dataset being used for train/val
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optimizer: optimizer to use
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        self.save_hyperparameters()#

        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.model = resnet18(
            normalize=True,
            hidden_mlp=self.hidden_mlp,
            output_dim=self.feat_dim,
            nmb_prototypes=self.nmb_prototypes,
            first_conv=self.first_conv,
            maxpool1=self.maxpool1
        )

        self.train_iters_per_epoch = self.num_samples // self.batch_size

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model.forward_backbone(x)

    def on_after_bacward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        unlabeled_batch = batch[0]
        batch = unlabeled_batch

        inputs, y = batch
        inputs = inputs[:-1]

        # normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)
        
        # multi-resolutional forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # swav loss
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id : bs * (crop_id + 1)]

                # get assigmnmets
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments_sinkhorn(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignmets prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v : bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
            loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.0}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def get_assignments_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=1024, type=int, help="hidden layer dimension in projection head") #1024
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        parser.add_argument(
            "--nmb_crops", type=int, default=[2, 4], nargs="+", help="list of number of crops (example: [2, 6])"
        )
        parser.add_argument(
            "--size_crops", type=int, default=[96, 36], nargs="+", help="crops resolutions (example: [224, 96])"
        )
        parser.add_argument(
            "--min_scale_crops",
            type=float,
            default=[0.33, 0.10],
            nargs="+",
            help="argument in RandomResizedCrop (example: [0.14, 0.05])",
        )
        parser.add_argument(
            "--max_scale_crops",
            type=float,
            default=[1, 0.33],
            nargs="+",
            help="argument in RandomResizedCrop (example: [1., 0.14])",
        )

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=15, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=64, type=int, help="batch size per gpu") #128

        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        # swav params
        parser.add_argument(
            "--crops_for_assign",
            type=int,
            nargs="+",
            default=[0, 1],
            help="list of crops id used for computing assignments",
        )
        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument(
            "--epsilon", default=0.05, type=float, help="regularization parameter for Sinkhorn-Knopp algorithm"
        )
        parser.add_argument(
            "--sinkhorn_iterations", default=3, type=int, help="number of iterations in Sinkhorn-Knopp algorithm"
        )
        parser.add_argument("--nmb_prototypes", default=512, type=int, help="number of prototypes")
        parser.add_argument(
            "--freeze_prototypes_epochs",
            default=1,
            type=int,
            help="freeze the prototypes during this many epochs from the start",
        )

        return parser