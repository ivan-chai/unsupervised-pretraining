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
    augmentations = []
    for _, augmentation in model_cfg.augmentations.items():
        augmentations.append(instantiate(augmentation))
    data = instantiate(cfg.datamodule, transform_list=augmentations)
    callbacks = []
    for _, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()
