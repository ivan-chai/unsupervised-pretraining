#!/usr/bin/env python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl


@hydra.main(config_path="../configs", config_name="evaluate_cfg.yaml")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    with open(cfg.model, "r") as fin:
        model_cfg = OmegaConf.load(fin)
    model = instantiate(model_cfg.model)
    train_transforms = []
    for _, augmentation in model_cfg.augmentations.train_transforms.items():
        train_transforms.append(instantiate(augmentation))
    test_transforms = []
    for _, augmentation in model_cfg.augmentations.test_transforms.items():
        test_transforms.append(instantiate(augmentation))
    data = instantiate(cfg.datamodule, train_transforms_list=train_transforms,
                       test_transforms_list=test_transforms)
    callbacks = []
    for _, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()
