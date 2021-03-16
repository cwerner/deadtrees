# train.py

import logging

# silence pytorch lightning bolts UserWarning about missing gym package (as of v0.3.0)
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pl_bolts
import pytorch_lightning as pl
import torch
from deadtrees.callbacks.checkpoint import checkpoint_callback
from deadtrees.loss.tversky.binary import BinaryTverskyLossV2
from deadtrees.utils import get_env, load_envs
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

warnings.simplefilter(action="ignore", category=UserWarning)


logger = logging.getLogger(__name__)


# Load environment variables
load_envs()

TILE_SIZE = 512


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> pl.Trainer:
    print(cfg)
    logger.info(f"PATH: {Path.cwd()}")
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    network = instantiate(cfg.network, cfg.train)
    data = instantiate(cfg.data)
    data.setup(
        data_dir=get_env("TRAIN_DATASET_PATH"),
        reduce=5000,
    )

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = pl.Trainer(
        **cfg.pl_trainer,
        logger=trainer_logger,
        gpus=1,
        precision=16,
        # auto_lr_find=True,
        max_epochs=30,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(network, data)
    if cfg.train.run_test:
        trainer.test(datamodule=data)

    return trainer


if __name__ == "__main__":
    main()
