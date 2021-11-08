from pathlib import Path

import pytest

from src.model.classification import ClassificationModel


@pytest.fixture(scope="session")
def data_path() -> Path:

    return Path("./data")


# TODO: сделать фикстуры для каждой модели. Вследствие их отсутствия тут resnet18
@pytest.fixture(scope="session")
def classification_model() -> ClassificationModel:

    model = ClassificationModel()

    return model
