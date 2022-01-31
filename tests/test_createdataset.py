import gzip
import io
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

import numpy as np
import rioxarray
from PIL import Image
from scripts.createdataset import _split_tile, Extractor, split_tiles

Path.ls = lambda x: list(x.iterdir())


@pytest.fixture
def sample_image():
    test_image = Path("tests") / "testdata" / "ortho_ms_2019_EPSG3044_092_011.tif.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        with gzip.open(test_image, "rb") as f_in:
            with open(Path(tmpdir) / test_image.stem, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            with rioxarray.open_rasterio(
                Path(tmpdir) / test_image.stem, chunks={"band": 4, "x": 512, "y": 512}
            ) as t:
                return t.persist()


@pytest.fixture
def sample_mask():
    test_mask = (
        Path("tests") / "testdata" / "ortho_ms_2019_EPSG3044_092_011_mask.tif.gz"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with gzip.open(test_mask, "rb") as f_in:
            with open(Path(tmpdir) / test_mask.stem, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            with rioxarray.open_rasterio(
                Path(tmpdir) / test_mask.stem, chunks={"band": 1, "x": 512, "y": 512}
            ) as t:
                return t.persist()


@pytest.fixture
def extractor():
    return Extractor(tile_size=256, source_dim=2048)


def test_extractor_substile_shapes(extractor, sample_image, sample_mask):
    assert extractor(sample_image, n_bands=4).shape == (64, 4, 256, 256)
    assert extractor(sample_mask, n_bands=1).shape == (64, 1, 256, 256)


def split_tile_subroutine():
    test_image = Path("tests") / "testdata" / "ortho_ms_2019_EPSG3044_092_011.tif.gz"
    test_mask = (
        Path("tests") / "testdata" / "ortho_ms_2019_EPSG3044_092_011_mask.tif.gz"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with gzip.open(test_image, "rb") as f_in_i, gzip.open(
            test_mask, "rb"
        ) as f_in_m:

            with open(Path(tmpdir) / test_image.stem, "wb") as fout_image:
                shutil.copyfileobj(f_in_i, fout_image)

            with open(Path(tmpdir) / test_mask.stem, "wb") as fout_mask:
                shutil.copyfileobj(f_in_m, fout_mask)

        img_file = Path(tmpdir) / test_image.stem
        msk_file = Path(tmpdir) / test_mask.stem
        samples = _split_tile(
            img_file, msk_file, source_dim=2048, tile_size=256, format="TIFF"
        )
    return samples


class TestSplitTile:

    samples = split_tile_subroutine()

    @property
    def sample_mask(self):
        raw_image = self.samples[0]["mask.tif"]
        return Image.open(io.BytesIO(raw_image))

    @property
    def sample_image(self):
        raw_image = self.samples[0]["rgbn.tif"]
        return Image.open(io.BytesIO(raw_image))

    def test_all_shard_keys_present(self):
        c = Counter()
        for s in self.samples:
            for k in list(set(s)):
                c[k] += 1
        assert c["__key__"] == c["rgbn.tif"] == c["mask.tif"] == c["txt"] == 64

    def test_shard_rgbn_shape(self):
        raw_image = self.samples[0]["rgbn.tif"]

        image = Image.open(io.BytesIO(raw_image))
        assert np.rollaxis(np.asarray(image), 2, 0).shape == (4, 256, 256)

    def test_shard_rgbn_filetype(self):
        raw_image = self.samples[0]["rgbn.tif"]

        image = Image.open(io.BytesIO(raw_image))
        assert image.format == "TIFF"

    def test_shard_mask_shape(self):
        assert np.asarray(self.sample_mask).shape == (256, 256)

    def test_shard_mask_filetype(self):
        assert self.sample_image.format == "TIFF"

    def test_shard_mask_valid_values(self):
        """Assert that mask values are in {0,1,2}"""
        raw_image = self.samples[0]["mask.tif"]

        image = Image.open(io.BytesIO(raw_image))
        values = set(np.asarray(image).ravel())
        assert values.issubset({0, 1, 2}) is True

    def test_shard_txt_matches_mask(self):
        """Assert that txt stats match actual mask values"""
        raw_image = self.samples[0]["mask.tif"]
        mask = np.asarray(Image.open(io.BytesIO(raw_image)))
        px_count = np.ones_like(mask)[mask > 0].sum()
        mask_frac = (px_count / mask.size) * 100
        frac = float(self.samples[0]["txt"])
        assert pytest.approx(frac, abs=1e-2) == mask_frac
