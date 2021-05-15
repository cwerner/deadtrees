import logging
from pathlib import Path
from warnings import warn

from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


# default used by the Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath=Path().cwd(),
    save_last=True,
    verbose=True,
    monitor="val/total_loss",
    mode="min",
)
