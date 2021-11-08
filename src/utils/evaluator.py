from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision import transforms


class Evaluator:

    def __init__(self, batch_size: int = 32, device="cuda"):

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._dataset = STL10("./data", split="test", transform=self.transforms)
        self._loader = DataLoader(self._dataset, batch_size=batch_size, shuffle=False)
        self.device = device

    def evaluate(self, model):
        model.to(self.device).eval()
        correct = 0

        for batch_idx, (image, labels) in enumerate(self._loader):
            image = image.to(self.device)
            logits = model(image)
            preds = logits.data.max(1)[1]

            correct += preds.cpu().eq(labels.data).sum()

        accuracy = correct / len(self._dataset)

        return accuracy
