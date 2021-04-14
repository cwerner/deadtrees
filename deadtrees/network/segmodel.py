# source: https://github.com/PyTorchLightning/pytorch-lightning-bolts (Apache2)

import logging
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from deadtrees.visualization.helper import show
from omegaconf import DictConfig
from pl_bolts.models.vision.unet import UNet
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class SemSegment(UNet, pl.LightningModule):  # type: ignore
    def __init__(
        self,
        train_conf: DictConfig,
        network_conf: DictConfig,
    ):
        super().__init__(**network_conf)
        self.save_hyperparameters()  # type: ignore

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()

        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        # log_dict = {"train_loss": loss_val}
        self.log("loss", loss_val)

        # return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)

        print(out.shape)
        print(mask.shape)

        loss_val = F.cross_entropy(out, mask, ignore_index=250)

        if batch_idx == 0:
            stats = batch["stats"] if "stats" in batch else None
            sample_chart = show(
                x=img.cpu(), y=mask.cpu(), y_hat=out.cpu(), n_samples=4, stats=stats
            )

            # TODO: use native wandb semantic segmentation UI

            # class_labels = {
            #     1: "deadtrees",
            # }

            # img = np.array(img[0].cpu().permute(1, 2, 0) * 255, dtype='uint8')
            # msk = np.array(mask[0].squeeze().cpu())
            # prd = np.array(out[0].squeeze().cpu())

            # logger.info(prd.shape)

            # xxx = wandb.Image(img, masks={
            #     "predictions": {
            #         "mask_data": prd,
            #         "class_labels": class_labels
            #         },
            #     "ground_truth": {
            #         "mask_data": msk,
            #         "class_labels": class_labels
            #         },
            #     },
            # )
            # self.logger.experiment.log({'native': xxx})

            self.logger.experiment.log(
                {
                    "sample": wandb.Image(
                        sample_chart, caption=f"Sample-{self.trainer.global_step}"
                    )
                },
                commit=False,
            )

        self.log("val_loss", loss_val)

        return loss_val

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train_conf.learning_rate,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


# class SemSegment(pl.LightningModule):

#     def __init__(
#         self,
#         lr: float = 0.01,
#         num_classes: int = 19,
#         num_layers: int = 5,
#         features_start: int = 64,
#         bilinear: bool = False
#     ):
#         """
#         Basic model for semantic segmentation. Uses UNet architecture by default.

#         The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
#         you will first need to download the KITTI dataset yourself. You can download the dataset `here.
#         <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

#         Implemented by:

#             - `Annika Brundyn <https://github.com/annikabrundyn>`_

#         Args:
#             num_layers: number of layers in each side of U-net (default 5)
#             features_start: number of features in first layer (default 64)
#             bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
#             lr: learning (default 0.01)
#         """
#         super().__init__()

#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.features_start = features_start
#         self.bilinear = bilinear
#         self.lr = lr

#         self.net = UNet(
#             num_classes=num_classes,
#             num_layers=self.num_layers,
#             features_start=self.features_start,
#             bilinear=self.bilinear
#         )

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_nb):
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index=250)
#         log_dict = {'train_loss': loss_val}
#         return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

#     def validation_step(self, batch, batch_idx):
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index=250)
#         return {'val_loss': loss_val}

#     def validation_epoch_end(self, outputs):
#         loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
#         log_dict = {'val_loss': loss_val}
#         return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
#         return [opt], [sch]

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = ArgumentParser(parents=[parent_parser], add_help=False)
#         parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
#         parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
#         parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
#         parser.add_argument(
#             "--bilinear",
#             action='store_true',
#             default=False,
#             help="whether to use bilinear interpolation or transposed"
#         )

#         return parser


# def cli_main():
#     from pl_bolts.datamodules import KittiDataModule

#     pl.seed_everything(1234)

#     parser = ArgumentParser()
#     # trainer args
#     parser = pl.Trainer.add_argparse_args(parser)
#     # model args
#     parser = SemSegment.add_model_specific_args(parser)
#     # datamodule args
#     parser = KittiDataModule.add_argparse_args(parser)

#     args = parser.parse_args()

#     # data
#     dm = KittiDataModule(args.data_dir).from_argparse_args(args)

#     # model
#     model = SemSegment(**args.__dict__)

#     # train
#     trainer = pl.Trainer().from_argparse_args(args)
#     trainer.fit(model, datamodule=dm)


# if __name__ == '__main__':
#     cli_main()
