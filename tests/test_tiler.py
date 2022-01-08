import tempfile
from math import prod
from pathlib import Path
from typing import Tuple, Union

import pytest
from attr import dataclass

import numpy as np
import rioxarray
from src.deployment.tiler import (
    divisible_without_remainder,
    inspect_tile,
    make_blocks_vectorized,
    Tiler,
    unmake_blocks_vectorized,
)


@pytest.fixture
def tiler():
    return Tiler()


@dataclass
class TileData:
    filename: Union[str, Path]
    size: Tuple[int, int]
    subtiles: Tuple[int, int]


example1 = TileData(
    Path("tests/testdata/tiles/ortho_2019_ESPG3044_49_11.tif"),
    (8192, 8192),
    (16, 16),
)

example2 = TileData(
    Path("tests/testdata/tiles/ortho_2019_ESPG3044_27_37.tif"),
    (8192, 7433),
    (16, 15),
)

example3 = TileData(
    Path("tests/testdata/tiles/ortho_2019_ESPG3044_52_26.tif"),
    (2649, 8192),
    (6, 16),
)

examples = [example1, example2, example3]


@pytest.mark.parametrize("a,b,result", [(10, 2, True), (5, 4, False), (2, 0, False)])
def test_divisible_without_remainder(a, b, result):
    assert divisible_without_remainder(a, b) == result


class TestBlocksVectorized:
    source = np.array([np.arange(16).reshape(4, 4)] * 3)
    target = np.array(
        [
            [[[0, 1], [4, 5]], [[0, 1], [4, 5]], [[0, 1], [4, 5]]],
            [[[2, 3], [6, 7]], [[2, 3], [6, 7]], [[2, 3], [6, 7]]],
            [[[8, 9], [12, 13]], [[8, 9], [12, 13]], [[8, 9], [12, 13]]],
            [[[10, 11], [14, 15]], [[10, 11], [14, 15]], [[10, 11], [14, 15]]],
        ]
    )

    def test_make_blocks_vectorized(self):
        """break tile into batch of subtiles"""
        np.testing.assert_array_equal(
            make_blocks_vectorized(self.source, 2), self.target
        )

    def test_unmake_blocks_vectorized(self):
        """place batches back into tile (2d)"""
        np.testing.assert_array_equal(
            unmake_blocks_vectorized(self.target[:, 0, :, :], 2, 4, 4), self.source[0]
        )


@pytest.mark.parametrize("tile", examples)
def test_tiler_inspect_tile_size(tile):
    assert inspect_tile(tile.filename).size == tile.size


@pytest.mark.parametrize("tile", examples)
def test_tiler_inspect_tile_subtiles(tile):
    assert inspect_tile(tile.filename).subtiles == tile.subtiles


@pytest.mark.parametrize("tile", examples[0:1])
def test_tiler_inspect_tile_subtile_not_divisible(tile):
    with pytest.raises(ValueError):
        inspect_tile(tile.filename, subtile_shape=(512, 211))


@pytest.mark.parametrize(
    "tile",
    [
        str(example1.filename),
        example1.filename,
        rioxarray.open_rasterio(example1.filename).sel(band=1, drop=True),
    ],
)
def test_tiler_infile_types(tile):
    assert inspect_tile(tile).size == example1.size


def test_tiler_catch_bad_subtile_dims():
    with pytest.raises(ValueError):
        Tiler(example1.filename, tile_shape=(8192, 8192), subtile_shape=(256, 250))


@pytest.mark.parametrize("tile", examples)
def test_tiler_load_file_subtiles_to_use(tiler, tile):
    tiler.load_file(tile.filename)
    assert sum(tiler._subtiles_to_use) == prod(tile.subtiles)


@pytest.mark.parametrize("tile", examples)
def test_tiler_get_batches(tiler, tile):
    tiler.load_file(tile.filename)
    assert tiler.get_batches().shape == (prod(tile.subtiles), 3, 512, 512)


@pytest.mark.parametrize("tile", examples)
def test_tiler_put_batches(tiler, tile):
    tiler.load_file(tile.filename)
    batches = tiler.get_batches()
    pred_batches = np.random.choice(
        a=[1, 0], size=(len(batches), 512, 512), p=[0.1, 0.9]
    )  # single layer
    tiler.put_batches(pred_batches)
    assert tiler._outdata.shape == (8192, 8192)


@pytest.mark.parametrize("tile", examples)
def test_tiler_write_file(tiler, tile):
    tiler.load_file(tile.filename)
    batches = tiler.get_batches()
    pred_batches = np.random.choice(
        a=[1, 0], size=(len(batches), 512, 512), p=[0.1, 0.9]
    )  # single layer
    tiler.put_batches(pred_batches)

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tiler.write_file(tmp.name)

        assert rioxarray.open_rasterio(tmp.name).values.shape == (1, *tile.size)
