# train.py

import logging

# silence pytorch lightning bolts UserWarning about missing gym package (as of v0.3.0)
import warnings
from pathlib import Path
from typing import List

import hydra
import torch
from deadtrees.callbacks.checkpoint import checkpoint_callback
from deadtrees.utils.env import get_env, load_envs
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

warnings.simplefilter(action="ignore", category=UserWarning)


log = logging.getLogger(__name__)


# TODO: Check why this is necessary!
torch.backends.cudnn.benchmark = False

# Load environment variables
load_envs()

TILE_SIZE = 512


@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig) -> Trainer:

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from deadtrees.utils import template_utils

    log.info(f"PATH: {Path.cwd()}")

    template_utils.extras(config)

    # Init Lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        data_dir=get_env("TRAIN_DATASET_PATH"),
        pattern=config.datamodule.pattern,
        pattern_extra=config.datamodule.get("pattern_extra", None),
        batch_size_extra=config.datamodule.get("batch_size_extra", None),
    )
    datamodule.setup()

    # Init Lightning model
    model: LightningModule = hydra.utils.instantiate(config.model, config.train)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for key, lg_conf in config["logger"].items():
            print(f"{_} ; {lg_conf}")
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info("Instantiating trainer")  # <{config.trainer._target_}>")

    # use later when we migrate to run/train scheme
    # trainer: Trainer = hydra.utils.instantiate(
    #     config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    # )

    trainer: Trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if config.train.run_test:
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric:
    #     return trainer.callback_metrics[optimized_metric]


if __name__ == "__main__":
    main()
