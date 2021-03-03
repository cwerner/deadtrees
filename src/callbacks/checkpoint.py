import logging
from pathlib import Path
from warnings import warn

import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch._C import device
from typing_extensions import Annotated

import wandb

logger = logging.getLogger(__name__)


# default used by the Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath=Path().cwd(),
    save_last=True,
    verbose=True,
    monitor="val_loss",
    mode="min",
    prefix="",
)


# try:
#     import torchvision
# except ImportError:
#     warn(
#         "You want to use `torchvision` which is not installed yet,"  # pragma: no-cover
#         " install it with `pip install torchvision`."
#     )


# class WandbImageSampler(Callback):
#     def __init__(
#         self,
#         num_samples: int = 3,
#         logging_batch_interval: int = 20,
#     ):
#         """
#         Evaluate images from given dataloader set.
#         """
#         super().__init__()
#         self.num_samples = num_samples
#         self.logging_batch_interval = logging_batch_interval

#     def _compose_images(self, images, trainer, pl_module):
#         if len(images.size()) == 2:
#             img_dim = pl_module.img_dim
#             images = images.view(self.num_samples, *img_dim)
#         grid = torchvision.utils.make_grid(images)
#         ndarr = (
#             grid.mul(255)
#             .add_(0.5)
#             .clamp_(0, 255)
#             .permute(1, 2, 0)
#             .to("cpu", torch.uint8)
#             .numpy()
#         )
#         im = Image.fromarray(ndarr)

#         if self.annotate:
#             label = f"Epoch: {trainer.current_epoch:003d}"
#             d = ImageDraw.Draw(im)
#             fnt = ImageFont.load_default()
#             x, _ = im.size
#             y = 5

#             # iteration label
#             w, h = fnt.getsize(label)
#             d.rectangle((x - w - 4, y, x - 2, y + h), fill="black")
#             d.text((x - w - 2, y), label, fnt=fnt, fill=(255, 255, 0))

#         return im

#     def on_train_batch_end(
#         self,
#         trainer,
#         pl_module,
#         outputs,
#         batch,
#         batch_idx,
#         dataloader_idx,
#     ):

#         if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
#             return

#         # generate images
#         pl_module.eval()

#         sample = next(iter(trainer.datamodule.test_data))

#         img, mask, stats = sample["image"], sample["mask"], sample["stats"]

#         predictions = pl_module(img.float())
#         pl_module.train()

#         images = self._compose_images(predictions, trainer, pl_module)
#         str_title = f"{pl_module.__class__.__name__}_images"

#         trainer.logger.experiment.log(
#             {"examples": [wandb.Image(images, caption=str_title)]}
#         )
