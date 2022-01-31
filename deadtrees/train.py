# train.py

# silence pytorch lightning bolts UserWarning about missing gym package (as of v0.3.0)
import warnings
from pathlib import Path
from typing import List, Optional

import hydra
from deadtrees.utils import utils
from deadtrees.utils.env import get_env
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    seed_everything,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase

warnings.simplefilter(action="ignore", category=UserWarning)


log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule

    ddir = Path(get_env("TRAIN_DATASET_PATH"))
    subfolders = ["train", "val", "test"]
    if all([(ddir / d).is_dir() for d in subfolders]):
        # dataset/train, dataset/val, dataset/test layout
        log.info(
            f"Instantiating datamodule <{config.datamodule._target_}> with train, val, test folder layout"
        )
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.datamodule,
            data_dir=[str(ddir / d) for d in subfolders],
            pattern=config.datamodule.pattern,
            pattern_extra=config.datamodule.get("pattern_extra", None),
            batch_size_extra=config.datamodule.get("batch_size_extra", None),
        )
    else:
        log.info(
            f"Instantiating datamodule <{config.datamodule._target_}> with single folder layout"
        )
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.datamodule,
            data_dir=get_env("TRAIN_DATASET_PATH"),
            pattern=config.datamodule.pattern,
            pattern_extra=config.datamodule.get("pattern_extra", None),
            batch_size_extra=config.datamodule.get("batch_size_extra", None),
        )
    datamodule.setup(
        in_channels=config.model.network.in_channels,
        classes=config.model.network.classes,
    )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
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

    # Get metric score for hyperparameter optimization
    score = trainer.callback_metrics.get(config.get("optimized_metric"))

    # Evaluate model on test set after training (using best model)
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score


if __name__ == "__main__":
    train()
