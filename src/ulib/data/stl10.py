from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import STL10


class STL10DataModule(LightningDataModule):

    def __init__(self, data_dir: str = "./dataset", batch_size: int = 32, num_workers: int = 4):

        super(STL10DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dims = (3, 96, 96)
        self.num_classes = 10

    def prepare_data(self) -> None:

        STL10(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit":
            train_full = STL10(self.data_dir, split="train", transform=self.transform)
            self.stl_train, self.stl_val = random_split(train_full, [4500, 500])

        if stage == "test":
            self.stl_test = STL10(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.stl_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.stl_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.stl_test, batch_size=self.batch_size, num_workers=self.num_workers)
