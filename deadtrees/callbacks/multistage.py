from typing import Optional

import torch
from deadtrees.utils import utils
from pytorch_lightning import Callback

log = utils.get_logger(__name__)


class MultiStage(Callback):
    def __init__(
        self,
        *,
        unfreeze_epoch: int,
        lr_reduce_epoch: Optional[int] = None,
        lr_reduce_fraction: Optional[float] = None,
    ):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch  # epoch when to unfreeze encoder
        self.lr_reduce_epoch = lr_reduce_epoch  # epoch when to reduce learning rate
        self.lr_reduce_fraction = lr_reduce_fraction  # reduce learning rate by fraction

    def on_train_epoch_start(self, trainer, pl_module):

        if trainer.current_epoch == 0:
            if pl_module.encoder_weights is None:
                log.error(
                    "No encoder weights given but MultiStage encoder freeze requested"
                )
                exit()
            else:
                log.info(
                    f"Using pre-trained encoder weights: {pl_module.encoder_weights}"
                )

            # freeze encoder
            log.info(f"NEW STAGE (epoch: {trainer.current_epoch}): Freeze encoder")
            pl_module.model.encoder.eval()
            for m in pl_module.model.encoder.modules():
                m.requires_grad_ = False

        if trainer.current_epoch == self.unfreeze_epoch:
            # unfreeze encoder, keep default learning rate
            log.info(f"NEW STAGE (epoch: {trainer.current_epoch}): Unfreeze encoder")
            pl_module.model.encoder.train()
            for m in pl_module.model.encoder.modules():
                m.requires_grad_ = True

        if self.lr_reduce_epoch:
            # we need a lr_reduce_fraction here!
            assert self.lr_reduce_fraction is not None

            if trainer.current_epoch == self.lr_reduce_epoch:
                # also use unfrozen encoder, lower learning rate
                log.info(
                    f"NEW STAGE (epoch: {trainer.current_epoch}): Lower LR rate by factor {self.lr_reduce_fraction}"
                )
                new_optimizer = torch.optim.Adam(
                    pl_module.parameters(),
                    lr=pl_module.hparams.training.learning_rate
                    / self.lr_reduce_fraction,
                )
                new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    new_optimizer, T_max=pl_module.hparams.training.cosineannealing_tmax
                )
                trainer.optimizers = [new_optimizer]
                trainer.lr_schedulers = trainer._configure_schedulers(
                    [new_scheduler], monitor=None, is_manual_optimization=False
                )
                trainer.optimizer_frequencies = (
                    []
                )  # or optimizers frequencies if you have any
