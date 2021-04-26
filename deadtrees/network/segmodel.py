# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging

import pytorch_lightning as pl
import torch
import wandb
from deadtrees.loss.tversky.binary import BinaryTverskyLossV2
from deadtrees.visualization.helper import show
from omegaconf import DictConfig
from pl_bolts.models.vision.unet import UNet
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# Taken from https://github.com/justusschock/dl-utils/blob/master/dlutils/metrics/dice.py
def binary_dice_coefficient(
    pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5, smooth: float = 1e-7
) -> torch.Tensor:
    """
    computes the dice coefficient for a binary segmentation task

    Args:
        pred: predicted segmentation (of shape Nx(Dx)HxW)
        gt: target segmentation (of shape NxCx(Dx)HxW)
        thresh: segmentation threshold
        smooth: smoothing value to avoid division by zero

    Returns:
        torch.Tensor: dice score
    """

    assert pred.shape == gt.shape

    pred_bool = pred > thresh

    intersec = (pred_bool * gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum() + gt.float().sum() + smooth)


class SemSegment(UNet, pl.LightningModule):  # type: ignore
    def __init__(
        self,
        train_conf: DictConfig,
        network_conf: DictConfig,
    ):
        super().__init__(**network_conf)
        self.save_hyperparameters()  # type: ignore

        self.binary_tversky_loss = BinaryTverskyLossV2(
            alpha=1.0 - train_conf["tversky_beta"],
            beta=train_conf["tversky_beta"],
        )
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def get_progress_bar_dict(self):
        """Hack to remove v_num from progressbar"""
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        img, mask, stats = batch
        img = img.float()
        mask = mask.long()
        pred = self(img)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)
        a_max = torch.argmax(softmaxed_pred, dim=1)

        # Calculate losses
        ce_loss = self.ce_loss(pred, mask)
        binary_tversky_loss = self.binary_tversky_loss(a_max, mask.unsqueeze(dim=1))
        total_loss = (ce_loss + binary_tversky_loss) / 2

        # Calculate dice coefficient
        dice_coeff = binary_dice_coefficient(a_max, mask)
        accuracy = (a_max == mask).float().mean()

        self.log("train/dice_coeff", dice_coeff, prog_bar=True)
        self.log("train/accuracy", accuracy, prog_bar=True)

        self.log("train/ce_loss", ce_loss)
        self.log("train/binary_tversky_loss", binary_tversky_loss)
        self.log("train/total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask, stats = batch
        img = img.float()
        mask = mask.long()
        pred = self(img)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)
        a_max = torch.argmax(softmaxed_pred, dim=1)

        # Calculate losses
        ce_loss = self.ce_loss(pred, mask)
        binary_tversky_loss = self.binary_tversky_loss(a_max, mask.unsqueeze(dim=1))
        total_loss = (ce_loss + binary_tversky_loss) / 2

        # Calculate dice coefficient
        dice_coeff = binary_dice_coefficient(a_max, mask)
        accuracy = (a_max == mask).float().mean()

        self.log("val/dice_coeff", dice_coeff)
        self.log("val/accuracy", accuracy, prog_bar=True)

        self.log("val/ce_loss", ce_loss)
        self.log("val/binary_tversky_loss", binary_tversky_loss)
        self.log("val/total_loss", total_loss)

        if batch_idx == 0:
            sample_chart = show(
                x=img.cpu(), y=mask.cpu(), y_hat=pred.cpu(), n_samples=4, stats=stats
            )
            for logger in self.logger:
                if isinstance(logger, pl.loggers.wandb.WandbLogger):
                    logger.experiment.log(
                        {
                            "sample": wandb.Image(
                                sample_chart,
                                caption=f"Sample-{self.trainer.global_step}",
                            )
                        },
                        commit=False,
                    )

        return total_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train_conf.learning_rate,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]
