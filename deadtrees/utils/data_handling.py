import math
from typing import List

import numpy as np
import pandas as pd


# source: https://stackoverflow.com/a/39430508/5300574
def make_blocks_vectorized(x: np.ndarray, d: int) -> np.ndarray:
    """Discet an array into subtiles
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


def split_df(
    df: pd.DataFrame,
    size: int,
    refcol: str = "frac",
) -> List[pd.DataFrame]:
    """
    Split dataset into train, val, test fractions while preserving
    the original mean ratio of deadtree pixels for all three buckets
    """

    df = df.sort_values(by=refcol, ascending=False).reset_index(drop=True)

    n_fractions = math.ceil(len(df) / size)
    fractions = [1 / n_fractions] * n_fractions
    all_fractions = sum(fractions)
    status = [0] * n_fractions

    df["class"] = -1

    idx = np.argmin(status)

    for rid, row in df.iterrows():
        idx = np.argmin(status)
        status[idx] += all_fractions / fractions[idx]
        df.loc[rid, "class"] = idx

    gdf = df.groupby("class")
    return [[f for f in gdf.get_group(x)["tile"]] for x in gdf.groups]
