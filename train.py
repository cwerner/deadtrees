# train.py

# silence pytorch lightning bilts UserWarning about missing gym package (as of v0.3.0)
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import logging
import hydra
from hydra.utils import instantiate

import numpy as np

import pandas as pd
from pathlib import Path

import pytorch_lightning as pl

import pl_bolts

from pathlib import Path
import torch

from omegaconf import DictConfig, OmegaConf

# import project specific files 
from src.loss.tversky.binary import BinaryTverskyLossV2

from src.callbacks.checkpoint import checkpoint_callback
from src.callbacks.checkpoint import WandbImageSampler

logger = logging.getLogger(__name__)


TILE_SIZE = 512
DATA_PATH = Path('data/dataset/train')


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> pl.Trainer:
    print(cfg)
    logger.info(f"PATH: {Path.cwd()}")
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    network = instantiate(cfg.network, cfg.train)
    data = instantiate(cfg.data)
    data.setup(data_dir=DATA_PATH, reduce=5000, )

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = pl.Trainer(
        **cfg.pl_trainer, 
        logger=trainer_logger,
        gpus=1, 
        precision=16, 
        #auto_lr_find=True, 
        max_epochs=30,
        checkpoint_callback=checkpoint_callback,
        )

    trainer.fit(network, data)
    if cfg.train.run_test:
        trainer.test(datamodule=data)

    return trainer

if __name__ == "__main__":
    main()
