# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import segmentation_models_pytorch as smp

import pandas as pd
import pytorch_lightning as pl
import torch
from deadtrees.loss.losses import (
    BoundaryLoss,
    class2one_hot,
    FocalLoss,
    GeneralizedDice,
)
from deadtrees.network.extra import EfficientUnetPlusPlus, ResUnet, ResUnetPlusPlus
from deadtrees.visualization.helper import show
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


def concat_extra(
    img: Tensor, mask: Tensor, distmap: Tensor, stats, *, extra
) -> Tuple[Tensor]:
    extra_imgs, extra_masks, extra_distmaps, extra_stats = list(zip(*extra))
    img = torch.cat((img, *extra_imgs), dim=0)
    mask = torch.cat((mask, *extra_masks), dim=0)
    distmap = torch.cat((distmap, *extra_distmaps), dim=0)
    stats.extend(sum(extra_stats, []))
    return img, mask, distmap, stats


def create_combined_batch(
    batch: Dict[str, Any]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    img, mask, distmap, stats = batch["main"]

    # grab extra datasets and concat tensors
    extra = [v for k, v in batch.items() if k.startswith("extra")]
    if extra:
        img, mask, distmap, stats = concat_extra(img, mask, distmap, stats, extra=extra)
    return img, mask, distmap, stats


class SemSegment(pl.LightningModule):  # type: ignore
    def __init__(
        self,
        train_conf: DictConfig,
        network_conf: DictConfig,
    ):
        super().__init__()

        architecture = network_conf.architecture.lower().strip()
        if architecture == "unet":
            Model = smp.Unet
        elif architecture in ["unetplusplus", "unet++"]:
            Model = smp.UnetPlusPlus
        elif architecture == "resunet":
            Model = ResUnet
        elif architecture in ["resunetplusplus", "resunet++"]:
            Model = ResUnetPlusPlus
        elif architecture in ["efficientunetplusplus", "efficientunet++"]:
            Model = EfficientUnetPlusPlus
        else:
            raise NotImplementedError(
                "Currently only Unet, ResUnet, Unet++, ResUnet++, and EfficientUnet++ architectures are supported"
            )

        # Model does not accept "architecture" as an argument, but we need to store it in hparams for inference
        # TODO: cleanup?
        clean_network_conf = network_conf.copy()
        del clean_network_conf.architecture
        del clean_network_conf.losses

        self.model = Model(**clean_network_conf)
        # self.model.apply(initialize_weights)

        self.save_hyperparameters()  # type: ignore

        self.classes = list(range(self.hparams["network_conf"]["classes"]))
        self.classes_wout_bg = [c for c in self.classes if c != 0]

        self.in_channels = self.hparams["network_conf"]["in_channels"]

        # losses
        self.generalized_dice_loss = None
        self.focal_loss = None
        self.boundary_loss = None

        # parse loss config
        self.initial_alpha = 0.01  # make this a hyperparameter and/ or scale with epoch
        self.boundary_loss_ramped = False
        for loss_component in network_conf.losses:
            if loss_component == "GDICE":
                # This the only required loss term
                self.generalized_dice_loss = GeneralizedDice(idc=self.classes_wout_bg)
            elif loss_component == "FOCAL":
                self.focal_loss = FocalLoss(idc=self.classes, gamma=2)
            elif loss_component == "BOUNDARY":
                self.boundary_loss = BoundaryLoss(idc=self.classes_wout_bg)
            elif loss_component == "BOUNDARY-RAMPED":
                self.boundary_loss = BoundaryLoss(idc=self.classes_wout_bg)
                self.boundary_loss_ramped = True
            else:
                raise NotImplementedError(
                    f"The loss component <{loss_component}> is not recognized"
                )

        logger.info(f"Losses: {network_conf.losses}")

        # checks: we require GDICE!
        assert self.generalized_dice_loss is not None

        self.dice_metric = smp.utils.metrics.Fscore(
            ignore_channels=[0],
        )

        self.dice_metric_with_bg = smp.utils.metrics.Fscore()

        self.stats = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }

    @property
    def alpha(self):
        """blending parameter for boundary loss - ramps from 0.01 to 0.99 in 0.01 steps by epoch"""
        return min((self.current_epoch + 1) * self.initial_alpha, 0.99)

    def get_progress_bar_dict(self):
        """Hack to remove v_num from progressbar"""
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    def calculate_loss(
        self, y_hat: Tensor, y: Tensor, stage: str, distmap: Optional[Tensor] = None
    ) -> Tensor:
        """calculate compound loss"""
        loss, loss_gd, loss_bd, loss_fo = 0, None, None, None

        if self.generalized_dice_loss:
            loss_gd = self.generalized_dice_loss(y_hat, y)
            self.log(f"{stage}/dice_loss", loss_gd)
            loss += loss_gd

        if self.boundary_loss:
            loss_bd = self.boundary_loss(y_hat, distmap)
            self.log(f"{stage}/boundary_loss", loss_bd)
            loss += self.alpha * loss_bd if self.boundary_loss_ramped else loss_bd

        if self.focal_loss:
            loss_fo = self.focal_loss(y_hat, y)
            self.log(f"{stage}/focal_loss", loss_fo)
            loss += loss_fo

        self.log(f"{stage}/total_loss", loss)

        return loss

    def log_metrics(self, y_hat: Tensor, y: Tensor, *, stage: str):
        dice_score = self.dice_metric(y_hat, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_hat, y)
        self.log(f"{stage}/dice", dice_score)
        self.log(f"{stage}/dice_with_bg", dice_score_with_bg)

    def training_step(self, batch, batch_idx):

        img, mask, distmap, stats = create_combined_batch(batch)

        logits = self.model(img)
        y = class2one_hot(mask, K=len(self.classes))
        y_hat = logits.softmax(dim=1)

        loss = self.calculate_loss(y_hat, y, "train", distmap=distmap)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warn("Train loss is NaN! What is going on?")
            return None

        self.log_metrics(y_hat, y, stage="train")

        # track training batch files
        self.stats["train"].update([x["file"] for x in stats])

        return loss

    def validation_step(self, batch, batch_idx):

        img, mask, distmap, stats = create_combined_batch(batch)

        logits = self.model(img)
        y = class2one_hot(mask, K=len(self.classes))
        y_hat = logits.softmax(dim=1)

        loss = self.calculate_loss(y_hat, y, stage="val", distmap=distmap)

        self.log_metrics(y_hat, y, stage="val")

        if batch_idx == 0:
            sample_chart = show(
                x=img.cpu(),
                y=mask.cpu(),
                y_hat=y_hat.cpu(),
                n_samples=min(img.shape[0], 8),
                stats=stats,
                dpi=72,
                display=False,
            )
            for logger in self.logger:
                if isinstance(logger, pl.loggers.wandb.WandbLogger):
                    import wandb

                    logger.experiment.log(
                        {
                            "sample": wandb.Image(
                                sample_chart,
                                caption=f"Sample-{self.trainer.global_step}",
                            )
                        },
                        commit=False,
                    )

        # track validation batch files
        self.stats["val"].update([x["file"] for x in stats])

        return loss

    def test_step(self, batch, batch_idx):
        img, mask, _, stats = batch

        logits = self.model(img)
        y = class2one_hot(mask, K=len(self.classes))
        y_hat = logits.softmax(dim=1)

        self.log_metrics(y_hat, y, stage="test")

        # track validation batch files
        self.stats["test"].update([x["file"] for x in stats])

    def teardown(self, stage=None) -> None:
        logger.debug(f"len(stats_train): {len(self.stats['train'])}")
        pd.DataFrame.from_records(
            list(dict(self.stats["train"]).items()), columns=["filename", "count"]
        ).to_csv("train_stats.csv", index=False)

        logger.debug(f"len(stats_val): {len(self.stats['val'])}")
        pd.DataFrame.from_records(
            list(dict(self.stats["val"]).items()), columns=["filename", "count"]
        ).to_csv("val_stats.csv", index=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train_conf.learning_rate,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


def initialize_weights(m):
    if getattr(m, "bias", None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
    for c in m.children():
        initialize_weights(c)
