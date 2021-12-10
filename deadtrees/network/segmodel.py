# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging
from collections import Counter

import segmentation_models_pytorch as smp

import pandas as pd
import pytorch_lightning as pl
import torch
from deadtrees.loss.losses import (
    BoundaryLoss,
    class2one_hot,
    CrossEntropy,
    GeneralizedDice,
    simplex,
)
from deadtrees.network.extra import EfficientUnetPlusPlus, ResUnet, ResUnetPlusPlus
from deadtrees.visualization.helper import show
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


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
        self.model = Model(**clean_network_conf)
        # self.model.apply(initialize_weights)

        self.save_hyperparameters()  # type: ignore

        self.classes = range(self.hparams["network_conf"]["classes"])
        self.in_channels = self.hparams["network_conf"]["in_channels"]

        # new losses
        self.alpha = 0.01  # nake this a hyperparameter and/ or scale with epoch
        classes_considered = list(range(1, len(self.classes)))
        self.generalized_dice_loss = GeneralizedDice(idc=classes_considered)
        self.boundary_loss = BoundaryLoss(idc=classes_considered)
        # self.ce_loss = CrossEntropy(idc=classes_considered)

        self.dice_metric = smp.utils.metrics.Fscore(
            ignore_channels=[0],
        )

        self.dice_metric_with_bg = smp.utils.metrics.Fscore()

        self.stats = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter(),
        }

    def get_progress_bar_dict(self):
        """Hack to remove v_num from progressbar"""
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    def _concat_extra(self, img, mask, distmap, stats, *, extra):
        extra_imgs, extra_masks, extra_distmaps, extra_stats = list(zip(*extra))
        img = torch.cat((img, *extra_imgs), dim=0)
        mask = torch.cat((mask, *extra_masks), dim=0)
        distmap = torch.cat((distmap, *extra_distmaps), dim=0)
        stats.extend(sum(extra_stats, []))
        return (img, mask, distmap, stats)

    def training_step(self, batch, batch_idx):
        img, mask, distmap, stats = batch["main"]

        # grab extra datasets and concat tensors
        extra = [v for k, v in batch.items() if k.startswith("extra")]
        if extra:
            img, mask, distmap, stats = self._concat_extra(
                img, mask, distmap, stats, extra=extra
            )

        # mask = mask.unsqueeze(1)
        pred = self.model(img)

        # loss_dice = self.criterion(pred, mask)
        # loss_focal = self.criterion2(pred, mask.squeeze(1))
        # loss = loss_dice * 0.5 + loss_focal * 0.5

        y = torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1)
        y_pred = pred.softmax(dim=1)

        loss_gd = self.generalized_dice_loss(y_pred, y)
        loss_bd = self.boundary_loss(y_pred, distmap)
        # loss_ce = self.ce_loss(y_pred, y)

        frac2 = min(self.alpha * (self.current_epoch + 1), 0.90)
        frac1 = 1 - frac2
        loss = frac1 * loss_gd + frac2 * loss_bd

        # TODO: simplify one-hot step
        dice_score = self.dice_metric(y_pred, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_pred, y)

        self.log("train/dice", dice_score)
        self.log("train/dice_with_bg", dice_score_with_bg)
        self.log("train/total_loss", loss)
        self.log("train/dice_loss", loss_gd)
        # self.log("train/ce_loss", loss_ce)
        self.log("train/boundary_loss", loss_bd)
        self.log("train/bdloss_frac", min(self.alpha * (self.current_epoch + 1), 0.90))

        # track training batch files
        self.stats["train"].update([x["file"] for x in stats])

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, distmap, stats = batch["main"]

        # grab extra datasets and concat tensors
        extra = [v for k, v in batch.items() if k.startswith("extra")]
        if extra:
            img, mask, distmap, stats = self._concat_extra(
                img, mask, distmap, stats, extra=extra
            )

        pred = self.model(img)

        # loss_dice = self.criterion(pred, mask.unsqueeze(1))
        # loss_focal = self.criterion2(pred, mask)  # .unsqueeze(1))
        # loss = loss_dice * 0.5 + loss_focal * 0.5

        y = torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1)
        y_pred = pred.softmax(dim=1)

        loss_gd = self.generalized_dice_loss(y_pred, y)
        loss_bd = self.boundary_loss(y_pred, distmap)
        # loss_ce = self.ce_loss(y_pred, y)

        frac2 = min(self.alpha * (self.current_epoch + 1), 0.90)
        frac1 = 1 - frac2

        loss = frac1 * loss_gd + frac2 * loss_bd

        # TODO: simplify one-hot step
        dice_score = self.dice_metric(y_pred, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_pred, y)

        self.log("val/dice", dice_score)
        self.log("val/dice_with_bg", dice_score_with_bg)
        self.log("val/total_loss", loss)
        self.log("val/dice_loss", loss_gd)
        # self.log("val/ce_loss", loss_ce)
        self.log("val/boundary_loss", loss_bd)

        if batch_idx == 0:
            sample_chart = show(
                x=img.cpu(),
                y=mask.cpu(),
                y_hat=pred.cpu(),
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
        img, mask, distmap, stats = batch
        pred = self.model(img)

        y_pred = pred.softmax(dim=1)
        y = torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1)

        # TODO: simplify one-hot step
        dice_score = self.dice_metric(y_pred, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_pred, y)

        self.log("test/dice", dice_score)
        self.log("test/dice_with_bg", dice_score_with_bg)

        # track validation batch files
        self.stats["test"].update([x["file"] for x in stats])

    def teardown(self, stage=None) -> None:
        print(f"len: {len(self.stats['train'])}")
        pd.DataFrame.from_records(
            list(dict(self.stats["train"]).items()), columns=["filename", "count"]
        ).to_csv("train_stats.csv", index=False)

        print(f"len: {len(self.stats['val'])}")
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
