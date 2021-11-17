from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import STL10

from unsupervised_pretraining.data.stl10albumentations import STL10Albumentations
import albumentations as A


class STL10DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform_list):
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
        self.transform = A.Compose(transform_list)

        self.num_classes = 10

    def prepare_data(self) -> None:
        STL10(self.data_dir, download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit":
            train_full = STL10Albumentations(self.data_dir, split="train", transform=self.transform)
            self.stl_train, self.stl_val = random_split(train_full, [4500, 500])
        if stage == "test":
            self.stl_test = STL10Albumentations(self.data_dir, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.stl_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.stl_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.stl_test, batch_size=self.batch_size, num_workers=self.num_workers)
