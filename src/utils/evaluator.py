from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchvision.datasets import STL10
from torchvision import transforms


class Evaluator:

    def __init__(self, batch_size: int = 32, device="cuda"):

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._train_dataset = STL10("./data", split="test", folds=1, transform=self.transforms)
        self._test_dataset = STL10("./data", split="test", transform=self.transforms)

        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._test_loader = DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False)

        self.device = device


    def train_fc(self, model, epochs=300):

        lr = 3e-4
        model.train()

        loss = nn.CrossEntropyLoss()
        opt = SGD(model.parameters(), lr=lr)

        for epoch in range(epochs):
            pass
        # here could be train loop




    def evaluate(self, model):

        model.to(self.device).eval()
        correct = 0

        for batch_idx, (image, labels) in enumerate(self._test_loader):
            image = image.to(self.device)
            logits = model(image)
            preds = logits.data.max(1)[1]

            correct += preds.cpu().eq(labels.data).sum()

        accuracy = correct / len(self._dataset)

        return accuracy
