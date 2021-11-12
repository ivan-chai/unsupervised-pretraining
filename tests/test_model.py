import torch

from src.models.classification import ClassificationModel


def test_model():
    # пока что здесь resnet18, у Влада эмбеддер всего лишь после двух эпох. Потом можно изменить.

    input_tensor = torch.ones([1, 3, 224, 224])

    model = ClassificationModel()
    model.eval()

    preds = model(input_tensor)

    assert preds.shape[1] == 10, f"incorrect FC-layer params"
