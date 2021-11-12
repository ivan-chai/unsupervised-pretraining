#!/usr/bin/env python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
import pytorch_lightning.callbacks.early_stopping


@hydra.main(config_path="../configs", config_name="train_cfg.yaml")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(42)

    data = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    callbacks = []
    for _, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))

    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()
