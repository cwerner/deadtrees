import io
import logging
import math
from functools import partial
from pathlib import Path
from typing import List, Optional

import albumentations as A
import webdataset as wds
from albumentations.pytorch import ToTensorV2

import numpy as np
import PIL
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# NOTE: mean ans std computed on train shards
class DeadtreeDatasetConfig:
    """Dataset configuration"""

    # 2019 tile images:
    #  Mean: [0.5699371099472046, 0.5824849605560303, 0.5583186745643616]
    #  STD: [0.3817858099937439, 0.3654171824455261, 0.3765888214111328]

    # 2017 tile images:
    #  Mean: [0.589081346988678, 0.6064110398292542, 0.594704806804657]
    #  STD: [0.37809664011001587, 0.3590240776538849, 0.3654339015483856]

    # 2017 + 2019
    #  Mean: [0.5795097351074219, 0.5944490432739258, 0.5765123963356018]
    #  STD: [0.38006964325904846, 0.36243298649787903, 0.3715078830718994]

    mean = np.array([0.5795097351074219, 0.5944490432739258, 0.5765123963356018])
    std = np.array([0.38006964325904846, 0.36243298649787903, 0.3715078830718994])
    tile_size = 512
    fractions = [0.7, 0.2, 0.1]


def split_shards(original_list, split_fractions):
    """Distribute shards into train/ valid/ test sets according to provided split ratios"""

    assert np.isclose(
        sum(split_fractions), 1.0
    ), f"Split fractions do not sum to 1: {sum(split_fractions)}"

    original_list = [str(x) for x in sorted(original_list)]

    sublists = []
    prev_index = 0
    for weight in split_fractions:
        next_index = prev_index + int(round((len(original_list) * weight), 0))
        sublists.append(original_list[prev_index:next_index])
        prev_index = next_index

    assert sum([len(x) for x in sublists]) == len(original_list), "Split size mismatch"

    if not all(len(x) > 0 for x in sublists):
        logger.warning("Unexpected shard distribution encountered - trying to fix this")
        if len(split_fractions) == 3:
            if len(sublists[0]) > 2:
                sublists[0] = original_list[:-2]
                sublists[1] = original_list[-2:-1]
                sublists[2] = original_list[-1:]
            else:
                raise ValueError(
                    f"Not enough shards (#{len(original_list)}) for new distribution"
                )

        elif len(split_fractions) == 2:
            sublists[0] = original_list[:-1]
            sublists[1] = original_list[-1:]
        else:
            raise ValueError
        logger.warning(f"New shard split: {sublists}")

    if len(sublists) != 3:
        logger.warning("No test shards specified")
        sublists.append(None)

    return sublists


def image_decoder(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("RGB")
    return np.asarray(img)


def mask_decoder(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("L")
    return np.asarray(img)


def sample_decoder(sample, img_suffix="rgb.png", msk_suffix="msk.png"):
    """Decode data triplet (image, mask stats) from sharded datastore"""

    assert img_suffix in sample, "Wrong image suffix provided"
    sample[img_suffix] = image_decoder(sample[img_suffix])
    if "txt" in sample:
        sample["txt"] = {"file": sample["__key__"], "frac": float(sample["txt"])}
    if msk_suffix in sample:
        sample[msk_suffix] = mask_decoder(sample[msk_suffix])
    return sample


def inv_normalize(x):
    return lambda x: x * DeadtreeDatasetConfig.std + DeadtreeDatasetConfig.mean


train_transform = A.Compose(
    [
        # A.Resize(256,256),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=DeadtreeDatasetConfig.mean, std=DeadtreeDatasetConfig.std),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        # A.Resize(256,256),
        A.Normalize(mean=DeadtreeDatasetConfig.mean, std=DeadtreeDatasetConfig.std),
        ToTensorV2(),
    ]
)


def transform(sample, transform_func=None):
    if transform_func:
        transformed = transform_func(
            image=sample["image"].copy(), mask=sample["mask"].copy()
        )
        sample["image"] = transformed["image"]
        sample["mask"] = transformed["mask"]
    return sample


class DeadtreesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        pattern,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.data_shards = sorted(Path(data_dir).glob(pattern))
        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()
        self.test_dataloader_conf = test_dataloader_conf or OmegaConf.create()

    def setup(
        self,
        split_fractions: List[float] = DeadtreeDatasetConfig.fractions,
    ) -> None:

        train_shards, valid_shards, test_shards = split_shards(
            self.data_shards, split_fractions
        )

        # determine the length of the dataset
        shard_size = sum(1 for _ in DataLoader(wds.WebDataset(train_shards[0])))
        logger.info(
            f"Shard size: {shard_size} (estimate base on file: {train_shards[0]})"
        )

        self.train_data = (
            wds.WebDataset(
                train_shards,
                length=len(train_shards)
                * shard_size
                // self.train_dataloader_conf["batch_size"],
            )
            .shuffle(shard_size)
            .map(sample_decoder)
            .rename(image="rgb.png", mask="msk.png", stats="txt")
            .map(partial(transform, transform_func=train_transform))
            .to_tuple("image", "mask", "stats")
        )

        self.val_data = (
            wds.WebDataset(
                valid_shards,
                length=len(valid_shards)
                * shard_size
                // self.val_dataloader_conf["batch_size"],
            )
            .shuffle(0)
            .map(sample_decoder)
            .rename(image="rgb.png", mask="msk.png", stats="txt")
            .map(partial(transform, transform_func=val_transform))
            .to_tuple("image", "mask", "stats")
        )

        if test_shards:
            self.test_data = (
                wds.WebDataset(
                    test_shards,
                    length=len(test_shards)
                    * shard_size
                    // self.test_dataloader_conf["batch_size"],
                )
                .shuffle(0)
                .map(sample_decoder)
                .rename(image="rgb.png", mask="msk.png", stats="txt")
                .map(partial(transform, transform_func=val_transform))
                .to_tuple("image", "mask", "stats")
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data.batched(
                self.train_dataloader_conf["batch_size"], partial=False
            ),
            batch_size=None,
            pin_memory=True,
            num_workers=self.train_dataloader_conf["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data.batched(
                self.val_dataloader_conf["batch_size"], partial=False
            ),
            batch_size=None,
            pin_memory=True,
            num_workers=self.val_dataloader_conf["num_workers"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data.batched(
                self.test_dataloader_conf["batch_size"], partial=False
            ),
            batch_size=None,
            pin_memory=True,
            num_workers=self.test_dataloader_conf["num_workers"],
        )
