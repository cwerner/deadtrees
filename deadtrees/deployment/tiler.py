# flake8: noqa: E402
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import xarray

warnings.filterwarnings("ignore", category=UserWarning)

import math
from dataclasses import dataclass

import numpy as np
import rioxarray


@dataclass
class TileInfo:
    size: Tuple[int, int]
    subtiles: Tuple[int, int]


def divisible_without_remainder(a, b):
    if b == 0:
        return False
    return True if a % b == 0 else False


def inspect_tile(
    infile: Union[str, Path, xarray.DataArray],
    tile_shape: Tuple[int, int] = (8192, 8192),
    subtile_shape: Tuple[int, int] = (512, 512),
) -> TileInfo:
    with rioxarray.open_rasterio(infile).sel(band=1, drop=True) if not isinstance(
        infile, xarray.DataArray
    ) else infile as da:

        shape = tuple(da.shape)

        if not divisible_without_remainder(tile_shape[0], subtile_shape[0]):
            raise ValueError(f"Shapes unaligned (v): {tile_shape[0], subtile_shape[0]}")

        if not divisible_without_remainder(tile_shape[1], subtile_shape[1]):
            raise ValueError(f"Shapes unaligned (h): {tile_shape[1], subtile_shape[1]}")

        subtiles = (
            math.ceil(shape[0] / subtile_shape[0]),
            math.ceil(shape[1] / subtile_shape[1]),
        )

    return TileInfo(size=shape, subtiles=subtiles)


class Tiler:
    def __init__(
        self,
        infile: Optional[Union[str, Path]] = None,
        tile_shape: Optional[Tuple[int, int]] = (2048, 2048),
        subtile_shape: Optional[Tuple[int, int]] = (256, 256),
    ) -> None:
        self._infile = infile
        self._tile_shape = tile_shape
        self._subtile_shape = subtile_shape

        if subtile_shape[0] != subtile_shape[1]:
            raise ValueError("Subtile required to have matching x/y dims")

        self._source: Optional[xarray.DataArray] = None
        self._target: Optional[xarray.DataArray] = None
        self._indata: Optional[np.ndarray] = None
        self._outdata: Optional[np.ndarray] = None
        self._batch_shape: Optional[np.ndarray] = None
        self._subtiles_to_use: Optional[np.ndarray] = None

        self._tile_info: Optional[TileInfo] = None

    def load_file(
        self,
        infile: Union[str, Path],
        tile_shape: Optional[Tuple[int, int]] = None,
        subtile_shape: Optional[Tuple[int, int]] = None,
    ) -> None:

        self._infile = infile
        self._tile_shape = tile_shape or self._tile_shape

        if subtile_shape:
            if subtile_shape[0] != subtile_shape[1]:
                raise ValueError("Subtile required to have matching x/y dims")
        self._subtile_shape = subtile_shape or self._subtile_shape

        self._tile_info = inspect_tile(
            self._infile, self._tile_shape, self._subtile_shape
        )

        self._source = rioxarray.open_rasterio(
            self._infile, chunks={"band": 4, "x": 256, "y": 256}
        )

        # define padded indata array and place original data inside
        sv = self._source.values
        if self._tile_shape != self._tile_info.size:
            self._indata = np.zeros((4, *self._tile_shape), dtype=self._source.dtype)
            self._indata[:, 0 : sv.shape[1], 0 : sv.shape[2]] = sv
        else:
            self._indata = sv

        # output xarray (single band)
        self._target = (
            self._source.sel(band=1, drop=True).astype("uint8").copy(deep=True)
        )

        # define padded outdata array
        self._outdata = np.zeros(self._tile_shape, dtype="uint8")

        # mark only necessary subtiles
        subtiles_mask = np.zeros(
            (
                self._tile_shape[0] // self._subtile_shape[0],
                self._tile_shape[1] // self._subtile_shape[1],
            ),
            dtype=bool,
        )
        subtiles_mask[
            0 : self._tile_info.subtiles[0], 0 : self._tile_info.subtiles[1]
        ] = 1
        self._subtiles_to_use = subtiles_mask.ravel()

    def write_file(self, outfile: Union[str, Path]) -> None:
        if self._target is not None:
            # copy data from outdata array into dataarray
            self._target[:] = self._outdata[
                0 : self._tile_info.size[0], 0 : self._tile_info.size[1]
            ]
            self._target.rio.to_raster(outfile, compress="LZW", tiled=True)

    def get_batches(self) -> np.ndarray:
        subtiles = make_blocks_vectorized(self._indata, self._subtile_shape[0])
        self._batch_shape = self._batch_shape or subtiles.shape
        return subtiles[self._subtiles_to_use]

    def put_batches(self, batches: np.ndarray) -> None:
        batches_expanded = []
        batch_idx = 0
        for flag in self._subtiles_to_use:
            if flag == 1:
                batches_expanded.append(batches[batch_idx])
                batch_idx += 1
            else:
                batches_expanded.append(np.zeros(batches[0].shape))

        batches_expanded = np.array(batches_expanded)

        self._outdata = unmake_blocks_vectorized(
            batches_expanded,
            self._subtile_shape[0],
            self._tile_shape[0],
            self._tile_shape[1],
        )

        # pass data into geo-registered rioxarray object (only subset of expanded tile if not complete tile)
        self._target = self._target.load()
        self._target.loc[:] = self._outdata[
            0 : self._tile_info.size[0], 0 : self._tile_info.size[1]
        ]


# https://stackoverflow.com/a/39430508/5300574
def make_blocks_vectorized(x: np.ndarray, d: int) -> np.ndarray:
    """Discet an array into subtiles: 3d -> 4d
    x: array (3d, channel:tile:tile)
    d: subtile width/ height
    """
    p, m, n = x.shape
    return (
        x.reshape(-1, m // d, d, n // d, d)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, p, d, d)
    )


def unmake_blocks_vectorized(x: np.ndarray, d: int, m: int, n: int) -> np.ndarray:
    """Merge subtiles back into array: 3d -> 2d
    x: array (3d, batch:subtile:subtile)
    d: subtile width/ height
    m: tile height
    n: tile width
    """
    return (
        np.concatenate(x)
        .reshape(m // d, n // d, d, d)
        .transpose(0, 2, 1, 3)
        .reshape(m, n)
    )
