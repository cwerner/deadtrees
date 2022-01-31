import io
import math
from functools import reduce

import pytest

import numpy as np
import pandas as pd
from deadtrees.utils.data_handling import split_df

TESTDATA = """tile,frac,status
ortho_ms_2019_EPSG3044_032_070_017,0.0,0
ortho_ms_2019_EPSG3044_032_070_018,0.0,0
ortho_ms_2019_EPSG3044_032_070_046,0.23,1
ortho_ms_2019_EPSG3044_032_070_047,0.58,1
ortho_ms_2019_EPSG3044_032_071_032,0.48,1
ortho_ms_2019_EPSG3044_032_071_033,0.01,1
ortho_ms_2019_EPSG3044_032_071_049,0.22,1
ortho_ms_2019_EPSG3044_032_071_050,0.29,1
ortho_ms_2019_EPSG3044_032_071_052,0.3,1
ortho_ms_2019_EPSG3044_032_071_053,0.4,1
ortho_ms_2019_EPSG3044_032_071_056,0.67,1
ortho_ms_2019_EPSG3044_032_071_057,0.39,1
ortho_ms_2019_EPSG3044_032_071_058,1.64,1
"""

eps = 1e-7

np.random.seed(42)


class TestSplitDf:
    # datasets to check
    data_fake = pd.DataFrame(
        {
            "tile": [f"fake_tile_{i:03d}.tif" for i in range(100)],
            "frac": np.random.gamma(9, 0.5, size=100) + eps,
            "status": np.ones(100, dtype=int),
        }
    )
    data_bad = pd.read_csv(io.StringIO(TESTDATA))
    data = pd.read_csv(io.StringIO(TESTDATA)).query("frac > 0")

    @pytest.mark.parametrize("n", [0, 100])
    def test_catch_invalid_size(self, n):
        with pytest.raises(ValueError):
            split_df(self.data, n)

    def test_catch_tiles_without_deadtrees(self):
        with pytest.raises(ValueError):
            split_df(self.data_bad, 3)

    def test_total_size_unchanged(self):
        result = split_df(self.data, 3)
        assert len(reduce(lambda z, y: z + y, result)) == len(self.data)

    def test_number_of_partitions_as_requested(self):
        result = split_df(self.data, 3)
        assert len(result) == math.ceil(len(self.data) / 3)

    def test_partitioned_totals_approx_equal(self):
        # dodgy, hand-crafted, and should be replaced by something rigid
        splits = split_df(self.data_fake, 10)
        totals = [
            self.data_fake[self.data_fake.tile.isin(s)].frac.sum() for s in splits
        ]
        assert [45] * len(totals) == pytest.approx(totals, abs=5)
