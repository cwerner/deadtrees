import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import skimage
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.transforms import (
    ToTensor,
)  # Rescale - rescale disabled for now/ messes with types

logger = logging.getLogger(__name__)


TILE_SIZE = 512
FRACTIONS = [0.7, 0.2, 0.1]
DF_FILE = "subtile_stats.csv"


def split_df(
    df: pd.DataFrame,
    fractions: List[float] = FRACTIONS,
    refcol: str = "deadtreepercent",
    no_zeros: bool = True,
    reduce: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Split dataset into train, val, test fractions while preserving
    the original mean ratio of deadtree pixels for all three buckets
    """

    all_fractions = sum(fractions)
    status = [0] * len(fractions)

    df = df.sort_values(by=refcol, ascending=False).reset_index(drop=True)

    if no_zeros:
        logger.warning("Don't use all-zero tiles")
        df = df[df[refcol] > 0.0]

    if reduce:
        logger.warning(f"Sample Data reduced to top {reduce} samples")

        df = df.head(reduce)

    df["class"] = -1

    idx = np.argmin(status)

    for rid, row in df.iterrows():
        idx = np.argmin(status)
        status[idx] += all_fractions / fractions[idx]
        df.loc[rid, "class"] = idx

    gdf = df.groupby("class")
    # DISABLED, since we want some tiles with actual deadtrees for inspection first
    # return [[f for f in gdf.get_group(x).sort_values(by='subtile')['subtile']] for x in gdf.groups]
    return [[f for f in gdf.get_group(x)["subtile"]] for x in gdf.groups]


class DeadtreeDataset(Dataset):
    """Deadtree dataset."""

    def __init__(
        self,
        files: List[Union[Path, str]],
        data_dir: Union[Path, str],
        transform: Optional[Any] = None,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.subtiles = pd.DataFrame({"filename": files})
        data_dir = to_absolute_path(str(data_dir))

        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.transform = transform

    def __len__(self):
        return len(self.subtiles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_dir / "image" / self.subtiles.iloc[idx, 0]
        img_mask_name = self.data_dir / "mask" / self.subtiles.iloc[idx, 0]

        image = skimage.io.imread(img_name)
        mask = skimage.io.imread(img_mask_name)

        stats = {
            "filename": img_name.name,
            "frac": round((float(mask.sum()) / mask.size) * 100, 2),
        }

        sample = {"image": image, "mask": mask, "stats": stats}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DeadtreesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        df_file,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        test_dataloader_conf: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.df_file = to_absolute_path(df_file)

        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create()
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create()
        self.test_dataloader_conf = test_dataloader_conf or OmegaConf.create()

        self.transform = transforms.Compose(
            [
                # Rescale(TILE_SIZE),
                ToTensor()
            ]
        )

    def setup(
        self,
        data_dir: Optional[Union[Path, str]] = None,
        df_file: Optional[pd.DataFrame] = None,
        split: List[float] = FRACTIONS,
        reduce: Optional[int] = None,
    ) -> None:
        DATA_PATH = data_dir or self.data_dir
        DF_FILE = df_file or self.df_file

        df = pd.read_csv(DF_FILE)
        train, val, test = split_df(df, split, reduce=reduce)

        self.train_data = DeadtreeDataset(train, DATA_PATH, transform=self.transform)
        self.valid_data = DeadtreeDataset(val, DATA_PATH, transform=self.transform)
        self.test_data = DeadtreeDataset(test, DATA_PATH, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, shuffle=True, **self.train_dataloader_conf)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, shuffle=False, **self.val_dataloader_conf)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, shuffle=False, **self.test_dataloader_conf)
