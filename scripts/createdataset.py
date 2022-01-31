import argparse
import io
import math
import random
import tarfile
import tempfile
from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil
import webdataset as wds

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from deadtrees.utils.data_handling import make_blocks_vectorized, split_df
from PIL import Image
from tqdm.contrib.concurrent import process_map

random.seed(42)

SHARDSIZE = 128
OVERSAMPLE_FACTOR = 2  # factor of random samples to dt + ndt samples

"""Summary:

This script builds up the final datasets for model training, validation and testing

The final datasets (combo) consists of three parts:
(1) tiles with classified deadtrees (frac > 0)
(2) tiles with non-deadtrees (aka healthy forest tiles, frac = 0)
(3) random other tiles (landuses: urban, arable, water etc., frac unknown, likely 0)

All shards are balanced to contain an equal amount of classified deadtree occurences to
allow a fair use of shards for traingin/ validation/ testing (composition of images should
be fair no matter the use)

Before the final dataset (combo) is created, various (temporary) datasets are created:
(1) train - raw, do not use
(2) train-balanced - balance the amounts of deadpixels (also filter to only include tiles with deadtrees)
(3) train-negsamples - a set of tiles that are guaranteed to contain non-dead trees
(4) train-randomsamples - a set of random other tiles

(X) combo: (2) + (4) [take turns]

"""


class Extractor:
    """Extract subtiles from rgbn or mask tile"""

    def __init__(self, *, tile_size: int = 256, source_dim: int = 2048):
        self.tile_size = tile_size
        self.source_dim = source_dim

    def __call__(self, t: Optional[xr.DataArray], *, n_bands: int):
        """Get data from tile, zeropad if necessary"""

        # default: all-zero in case no mask file exists
        data = np.zeros((n_bands, self.source_dim, self.source_dim), dtype=t.dtype)

        if t is not None:
            if (len(t.x) * len(t.y)) != (self.source_dim * self.source_dim):
                data[:, 0 : 0 + t.shape[1], 0 : 0 + t.shape[2]] = t.values
            else:
                data = t.values

        return make_blocks_vectorized(data, self.tile_size)


def _split_tile(
    image: Path,
    mask: Path,
    *,
    source_dim: int,
    tile_size: int,
    format: str,
    valid_subtiles: Optional[Iterable[str]] = None,
) -> List[Tuple[str, bytes, bytes]]:
    """Helper func for split_tiles"""

    extract = Extractor(tile_size=tile_size, source_dim=source_dim)

    n_bands = 4  # RGBN
    chunks = {"band": n_bands, "x": tile_size, "y": tile_size}
    with rioxarray.open_rasterio(image, chunks=chunks) as t:
        subtile_rgbn = extract(t, n_bands=n_bands)

    # process (optional) mask data
    n_bands = 1  # Bool
    if mask:
        chunks = {"band": n_bands, "x": tile_size, "y": tile_size}
        with rioxarray.open_rasterio(mask, chunks=chunks) as t:
            subtile_mask = extract(t, n_bands=n_bands)
    else:
        subtile_mask = extract(None, n_bands=n_bands)

    samples = []
    if format == "TIFF":
        suffix = "tif"
    elif format == "PNG":
        suffix = "png"
    else:
        raise NotImplementedError

    for i in range(subtile_rgbn.shape[0]):
        subtile_name = f"{image.stem}_{i:03}"

        if np.min(subtile_rgbn[i]) != np.max(subtile_rgbn[i]):
            im = Image.fromarray(np.rollaxis(subtile_rgbn[i], 0, 3), "RGBA")
            im_mask = Image.fromarray(subtile_mask[i].squeeze())

            im_byte_arr = io.BytesIO()
            im.save(im_byte_arr, format=format)
            im_byte_arr = im_byte_arr.getvalue()

            im_mask_byte_arr = io.BytesIO()
            im_mask.save(im_mask_byte_arr, format=format)
            im_mask_byte_arr = im_mask_byte_arr.getvalue()

            sample = {
                "__key__": subtile_name,
                f"rgbn.{suffix}": im_byte_arr,
                f"mask.{suffix}": im_mask_byte_arr,
                "txt": str(
                    round(
                        float(np.count_nonzero(subtile_mask[i]))
                        / (tile_size * tile_size)
                        * 100,
                        2,
                    )
                ),
            }
            if (valid_subtiles is None) or (subtile_name in valid_subtiles):
                samples.append(sample)
    return samples


def split_tiles(images, masks, workers: int, shardpattern: str, **kwargs) -> List[Any]:
    """Split tile into subtiles in parallel and save them to disk"""

    valid_subtiles = kwargs.get("valid_subtiles", None)

    stats = []
    with wds.ShardWriter(shardpattern, maxcount=SHARDSIZE) as sink:

        data = process_map(
            partial(_split_tile, **kwargs),
            images,
            masks,
            max_workers=workers,
            chunksize=1,
        )

        for sample in reduce(lambda z, y: z + y, data):
            if sample:
                if valid_subtiles:
                    if sample["__key__"] in valid_subtiles:
                        sink.write(sample)
                        stats.append((sample["__key__"], sample["txt"], "1"))
                else:
                    if float(sample["txt"]) > 0:
                        sink.write(sample)
                        stats.append((sample["__key__"], sample["txt"], "1"))
                    else:
                        # not included in shard
                        stats.append((sample["__key__"], sample["txt"], "0"))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("outdir", type=Path)

    num_cores = psutil.cpu_count(logical=False)
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=num_cores,
        help="number of workers for parallel execution [def: %(default)s]",
    )

    parser.add_argument(
        "--source_dim",
        dest="source_dim",
        type=int,
        default=2048,
        help="size of input tiles [def: %(default)s]",
    )

    parser.add_argument(
        "--tile_size",
        dest="tile_size",
        type=int,
        default=256,
        help="size of final tiles that are then passed to the model [def: %(default)s]",
    )

    parser.add_argument(
        "--format",
        dest="format",
        type=str,
        default="TIFF",
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

    parser.add_argument(
        "--stats",
        dest="stats_file",
        type=Path,
        default=Path("stats.csv"),
        help="use this file to record stats",
    )

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    Path(args.outdir / "train").mkdir(parents=True, exist_ok=True)

    if args.tmp_dir:
        print(f"Using custom tmp dir: {args.tmp_dir}")
        Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

    if args.format == "TIFF":
        suffix = "tif"
    elif args.format == "PNG":
        suffix = "png"
    else:
        raise NotImplementedError

    SHUFFLE = True  # shuffle subtile order within shards (with fixed seed)

    # subtile_stats = split_tiles(train_files)
    images = sorted(args.image_dir.glob("*.tif"))
    masks = sorted(args.mask_dir.glob("*.tif"))

    image_names = {i.name for i in images}
    mask_names = {i.name for i in masks}

    # limit set of images to images that have equivalent mask tiles
    train_images = [i for i in images if i.name in image_names.intersection(mask_names)]

    cfg = dict(
        source_dim=args.source_dim,
        tile_size=args.tile_size,
        format=args.format,
    )

    subtile_stats = split_tiles(
        train_images,
        masks,
        args.workers,
        str(args.outdir) + "/train/train-%06d.tar",
        **cfg,
    )

    with open(args.outdir / args.stats_file, "w") as fout:
        fout.write("tile,frac,status\n")
        for i, (fname, frac, status) in enumerate(subtile_stats):
            line = f"{fname},{frac},{status}\n"
            fout.write(line)

    # rebalance shards so we get similar distributions in all shards
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdir:
        print(f"Created a temporary directory: {tmpdir}")

        print("Extract source tars")
        # untar input
        for tf_name in sorted((args.outdir / "train").glob("train-00*.tar")):
            with tarfile.open(tf_name) as tf:
                tf.extractall(tmpdir)

        print("Write balanced shards from deadtree samples")
        df = pd.read_csv(args.outdir / args.stats_file)
        df = df[df.status > 0]
        n_valid = len(df)

        splits = split_df(df, SHARDSIZE)

        # drop incomplete shards
        splits = [x for x in splits if len(x) == SHARDSIZE]

        for s_cnt, s in enumerate(splits):

            with tarfile.open(
                args.outdir / "train" / f"train-balanced-{s_cnt:06}.tar", "w"
            ) as dst:

                if SHUFFLE:
                    random.shuffle(s)
                for i in s:
                    dst.add(f"{tmpdir}/{i}.mask.{suffix}", f"{i}.mask.{suffix}")
                    dst.add(f"{tmpdir}/{i}.rgbn.{suffix}", f"{i}.rgbn.{suffix}")
                    dst.add(f"{tmpdir}/{i}.txt", f"{i}.txt")

    # --------------------------------------------------------

    # check if negative samples folder exists
    mask_dir_ns = args.mask_dir.parent / (args.mask_dir.name + ".neg_sample")
    if mask_dir_ns.is_dir():
        masks_ns = sorted(mask_dir_ns.glob("*.tif"))
        mask_names_ns = {i.name for i in masks_ns}

        train_images_ns = [
            i for i in images if i.name in image_names.intersection(mask_names_ns)
        ]

        cfg = dict(
            source_dim=args.source_dim,
            tile_size=args.tile_size,
            format=args.format,
        )
        subtile_stats_ns = split_tiles(
            train_images_ns,
            masks_ns,
            args.workers,
            str(args.outdir) + "/train/train-negativesamples-%06d.tar",
            **cfg,
        )

        n_valid_ns = 0
        with open(args.outdir / "stats_ns.csv", "w") as fout:
            fout.write("tile,frac,status\n")
            for i, (fname, frac, status) in enumerate(subtile_stats_ns):
                # for negative samples we only write the actual subtiles out
                if float(frac) > 0:
                    line = f"{fname},{frac},{status}\n"
                    fout.write(line)
                    n_valid_ns += 1

        subtile_stats.extend(subtile_stats_ns)
        n_valid += n_valid_ns

    # create sets for random tile dataset
    # use all subtiles not covered in train or train-negatviesamples

    n_subtiles = (args.source_dim // args.tile_size) ** 2

    all_subtiles = []
    for image_name in image_names:
        all_subtiles.extend(
            [f"{Path(image_name).stem}_{c:03}" for c in range(n_subtiles)]
        )
    all_subtiles = set(all_subtiles)

    # rule of thumb: select 10x the number of dead-tree+negative samples (can be limited later)
    n_samples = n_valid * OVERSAMPLE_FACTOR
    random_subtiles = random.sample(
        tuple(all_subtiles - set([x[0] for x in subtile_stats if int(x[2]) == 1])),
        n_samples,
    )

    # the necessary tile to process
    random_tiles = sorted(list(set([x[:-4] for x in random_subtiles])))

    all_images = sorted(args.image_dir.glob("*.tif"))
    random_images = [x for x in all_images if x.stem in random_tiles]

    print("STATS")
    print(len(all_subtiles))
    print(len(subtile_stats))
    print(len(random_subtiles))
    print(len(random_images))

    cfg = dict(
        source_dim=args.source_dim,
        tile_size=args.tile_size,
        format=args.format,
        valid_subtiles=random_subtiles,  # subset data with random selection of subtiles
    )
    subtile_stats_rnd = split_tiles(
        random_images,
        [None] * len(random_images),
        args.workers,
        str(args.outdir) + "/train/train-randomsamples-%06d.tar",
        **cfg,
    )

    stats_file_rnd = Path(args.stats_file.stem + "_rnd.csv")
    with open(args.outdir / stats_file_rnd, "w") as fout:
        fout.write("tile,frac,status\n")
        for i, (fname, frac, status) in enumerate(subtile_stats_rnd):
            line = f"{fname},{frac},{status}\n"
            fout.write(line)

    # also create combo dataset
    # source A: train-balanced, source B: randomsample
    # NOTE: combo dataset has double the default shardsize (2*128), samples alternate between regular and random sample
    train_balanced_shards = [
        str(x) for x in sorted((args.outdir / "train").glob("train-balanced*"))
    ]
    train_balanced_shards_rnd = [
        str(x) for x in sorted((args.outdir / "train").glob("train-random*"))
    ]
    train_balanced_shards_rnd = train_balanced_shards_rnd[: len(train_balanced_shards)]

    shardpattern = str(args.outdir) + "/train/train-combo-%06d.tar"

    with wds.ShardWriter(shardpattern, maxcount=SHARDSIZE * 2) as sink:
        for shardA, shardB in zip(train_balanced_shards, train_balanced_shards_rnd):

            for sA, sB in zip(wds.WebDataset(shardA), wds.WebDataset(shardB)):
                sink.write(sA)
                sink.write(sB)


if __name__ == "__main__":
    main()
