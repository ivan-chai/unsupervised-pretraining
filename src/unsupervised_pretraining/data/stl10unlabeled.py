from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10

from unsupervised_pretraining.data.stl10albumentations import STL10Albumentations
import albumentations as A


class STL10DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, train_transforms_list, test_transforms_list):
        """Модуль данных для загрузки датасета STL10.
        Датасет STL-10 для экспериментов с обучением нейросетей без учителя (https://cs.stanford.edu/~acoates/stl10/).
        При первом вызове скачивает датасет в указанную в data_dir директорию.

        :param data_dir: директория для хранения датасета.
        :param batch_size: размер батча.
        :param num_workers: количество процессов для загрузки данных во время обучения/тестирования.
        """
        super(STL10DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = A.Compose(train_transforms_list)

    def prepare_data(self) -> None:
        STL10(self.data_dir, download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit":
            self.train_unlabeled = STL10Albumentations(self.data_dir, split="unlabeled", transform=self.train_transform)
            self.dummy_val_dataloader = STL10Albumentations(
                self.data_dir, split="train", transform=self.train_transform)
            self.dummy_val_dataloader, _ = random_split(self.dummy_val_dataloader, [100, 4900])

    def train_dataloader(self):
        return DataLoader(self.train_unlabeled, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dummy_val_dataloader, batch_size=self.batch_size, num_workers=self.num_workers)
