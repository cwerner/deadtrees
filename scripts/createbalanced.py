# createbalanced

import argparse
import logging
import math
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

import webdataset as wds

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_df(
    df: pd.DataFrame,
    size: int,
    refcol: str = "deadtreepercent",
    no_zeros: bool = False,
    reduce: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Split dataset into train, val, test fractions while preserving
    the original mean ratio of deadtree pixels for all three buckets
    """

    df = df.sort_values(by=refcol, ascending=False).reset_index(drop=True)

    if no_zeros:
        logger.warning("Don't use all-zero tiles")
        df = df[df[refcol] > 0.0]

    n_fractions = math.floor(len(df) / size)
    fractions = [1 / n_fractions] * n_fractions
    all_fractions = sum(fractions)
    status = [0] * n_fractions

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
    return [[f for f in gdf.get_group(x)["tile"]] for x in gdf.groups]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("balanced_dir", type=Path)
    parser.add_argument("balanced_short_dir", type=Path)

    parser.add_argument(
        "--balanced_size",
        dest="balanced_size",
        type=int,
        default=256,
        help="size of balanced shards [def: %(default)s]",
    )

    parser.add_argument(
        "--balanced_short_size",
        dest="balanced_short_size",
        type=int,
        default=64,
        help="size of balanced-short shards [def: %(default)s]",
    )

    parser.add_argument(
        "--format",
        dest="format",
        type=str,
        default="PNG",
        choices=["PNG", "TIFF"],
        help="target file format (PNG, TIFF) [def: %(default)s]",
    )

    parser.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        type=Path,
        default=None,
        help="use this location as tmp dir",
    )

    args = parser.parse_args()

    if args.format == "TIFF":
        suffix = "tif"
    elif args.format == "PNG":
        suffix = "png"
    else:
        raise NotImplementedError

    df = pd.read_csv(args.stats)

    if args.tmp_dir:
        print(f"Using custom tmp dir: {args.tmp_dir}")
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

    args.balanced_dir.mkdir(parents=True, exist_ok=True)
    args.balanced_short_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdir:
        # The context manager will automatically delete this directory after this section
        print(f"Created a temporary directory: {tmpdir}")

        print("Extract source tars")
        # untar input
        for tf_name in sorted(Path(args.input_dir).glob("train-00*.tar")):
            with tarfile.open(tf_name) as tf:
                tf.extractall(tmpdir)

        print("Write balanced shards")
        splits = split_df(df, args.balanced_size)
        for s_cnt, s in enumerate(splits):
            print(f"Split: {s_cnt}: {len(s)}, {len(splits)}")
            s = s[: args.balanced_size]

            with tarfile.open(
                args.balanced_dir / f"train-balanced-{s_cnt:06d}.tar", "w"
            ) as dst:
                for i in s:
                    dst.add(f"{tmpdir}/{i}.msk.{suffix}", f"{i}.msk.{suffix}")
                    dst.add(f"{tmpdir}/{i}.rgb.{suffix}", f"{i}.rgb.{suffix}")
                    dst.add(f"{tmpdir}/{i}.nir.{suffix}", f"{i}.nir.{suffix}")
                    dst.add(f"{tmpdir}/{i}.txt", f"{i}.txt")

        print("Write balanced-short shards")
        # --- short
        splits = split_df(df, args.balanced_short_size, no_zeros=True)
        for s_cnt, s in enumerate(splits):
            print(f"Split: {s_cnt}: {len(s)}, {len(splits)}")
            s = s[: args.balanced_short_size]

            with tarfile.open(
                args.balanced_short_dir / f"train-balanced-short-{s_cnt:06d}.tar", "w"
            ) as dst:
                for i in s:
                    dst.add(f"{tmpdir}/{i}.msk.{suffix}", f"{i}.msk.{suffix}")
                    dst.add(f"{tmpdir}/{i}.rgb.{suffix}", f"{i}.rgb.{suffix}")
                    dst.add(f"{tmpdir}/{i}.nir.{suffix}", f"{i}.nir.{suffix}")
                    dst.add(f"{tmpdir}/{i}.txt", f"{i}.txt")


if __name__ == "__main__":
    main()
