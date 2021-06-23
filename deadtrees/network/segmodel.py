# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

import pytorch_lightning as pl
import torch
from deadtrees.visualization.helper import show
from omegaconf import DictConfig
from pl_bolts.models.vision.unet import UNet

logger = logging.getLogger(__name__)


class SemSegment(UNet, pl.LightningModule):  # type: ignore
    def __init__(
        self,
        train_conf: DictConfig,
        network_conf: DictConfig,
    ):
        super().__init__(**network_conf)
        self.save_hyperparameters()  # type: ignore

        self.criterion = DiceCELoss(
            softmax=True,
            include_background=False,
            to_onehot_y=True,
        )

        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
        )

    def get_progress_bar_dict(self):
        """Hack to remove v_num from progressbar"""
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        img, mask, stats = batch
        img = img.float()
        mask = mask.long().unsqueeze(1)
        pred = self(img)

        loss = self.criterion(pred, mask)

        # TODO: simplify one-hot step
        dice_score, _ = self.dice_metric(
            y_pred=pred.softmax(dim=1),
            y=torch.zeros_like(pred).scatter_(1, mask, 1),
        )

        self.log("train/dice", dice_score)
        self.log("train/total_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, stats = batch
        img = img.float()
        mask = mask.long()
        pred = self(img)

        loss = self.criterion(pred, mask.unsqueeze(1))

        dice_score, _ = self.dice_metric(
            y_pred=pred.softmax(dim=1),
            y=torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1),
        )

        self.log("val/dice", dice_score)
        self.log("val/total_loss", loss)

        if batch_idx == 0:
            sample_chart = show(
                x=img.cpu(),
                y=mask.cpu(),
                y_hat=pred.cpu(),
                n_samples=4,
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

        return loss

    def test_step(self, batch, batch_idx):
        img, mask, stats = batch
        img = img.float()
        mask = mask.long()
        pred = self(img)

        dice_score, _ = self.dice_metric(
            y_pred=pred.softmax(dim=1),
            y=torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1),
        )

        self.log("test/dice", dice_score)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train_conf.learning_rate,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]
