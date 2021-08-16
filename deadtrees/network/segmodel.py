# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging
from collections import Counter

import segmentation_models_pytorch as smp

import pandas as pd
import pytorch_lightning as pl
import torch
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

        del network_conf.architecture
        self.model = Model(**network_conf)
        # self.model.apply(initialize_weights)
        self.save_hyperparameters()  # type: ignore

        # CHECK:
        # - softmax yes/ no ?
        # - incl_background yes/ no ?

        # self.criterion = DiceCELoss(
        #     softmax=True,
        #     include_background=False,
        #     to_onehot_y=True,
        # )

        # Reference:
        # Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning loss
        #    function for highly unbalanced segmentations. DLMIA 2017.
        # self.criterion = GeneralizedDiceLoss(
        #     softmax=True,
        #     include_background=False,
        #     to_onehot_y=True,
        # )

        # TODO: check if this is actually generalized dice loss???
        #       check if classes=[1] is correct
        self.criterion = smp.losses.DiceLoss(
            mode="multiclass",
            classes=[1],
            # log_loss=True,
        )

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

    def training_step(self, batch, batch_idx):
        img, mask, stats = batch["main"]

        # grab extra datasets and concat tensors
        extra_imgs, extra_masks, extra_stats = list(
            zip(*[v for k, v in batch.items() if k.startswith("extra")])
        )
        if len(extra_imgs) > 0:
            img = torch.cat((img, *extra_imgs), dim=0)
            mask = torch.cat((mask, *extra_masks), dim=0)
            stats.extend(sum(extra_stats, []))

        img = img.float()
        mask = mask.long().unsqueeze(1)
        pred = self.model(img)

        loss = self.criterion(pred, mask)

        y_pred = pred.softmax(dim=1)
        y = torch.zeros_like(pred).scatter_(1, mask, 1)

        # TODO: simplify one-hot step
        dice_score = self.dice_metric(y_pred, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_pred, y)

        self.log("train/dice", dice_score)
        self.log("train/dice_with_bg", dice_score_with_bg)
        self.log("train/total_loss", loss)

        # track training batch files
        self.stats["train"].update([x["file"] for x in stats])

        return loss

    def validation_step(self, batch, batch_idx):

        img, mask, stats = batch["main"]

        # grab extra datasets and concat tensors
        extra_imgs, extra_masks, extra_stats = list(
            zip(*[v for k, v in batch.items() if k.startswith("extra")])
        )
        if len(extra_imgs) > 0:
            img = torch.cat((img, *extra_imgs), dim=0)
            mask = torch.cat((mask, *extra_masks), dim=0)
            stats.extend(sum(extra_stats, []))

        img = img.float()
        mask = mask.long()
        pred = self.model(img)

        loss = self.criterion(pred, mask.unsqueeze(1))

        y_pred = pred.softmax(dim=1)
        y = torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1)

        # TODO: simplify one-hot step
        dice_score = self.dice_metric(y_pred, y)
        dice_score_with_bg = self.dice_metric_with_bg(y_pred, y)

        self.log("val/dice", dice_score)
        self.log("val/dice_with_bg", dice_score_with_bg)
        self.log("val/total_loss", loss)

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
        img, mask, stats = batch
        img = img.float()
        mask = mask.long()
        pred = self.model(img)

        y_pred = pred.softmax(dim=1)
        y = torch.zeros_like(pred).scatter_(1, mask.unsqueeze(1), 1)

        # TODO: simplify one-hot step
        (dice_score,) = self.dice_metric(y_pred, y)
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
