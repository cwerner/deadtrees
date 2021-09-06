# createbalanced

import argparse
import io
import logging
import math
import random
import tarfile
import tempfile
from functools import partial, reduce
from itertools import islice, tee
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import webdataset as wds

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

np.random.seed(1234)


logger = logging.getLogger(__name__)


def make_poly(coords: pd.Series) -> Polygon:
    """Create Shapely polygon (a tile boundary) from x1,y1 and x2,y2 coordinates"""
    xs = [coords[v] for v in "x1,x1,x2,x2,x1".split(",")]
    ys = [coords[v] for v in "y1,y2,y2,y1,y1".split(",")]
    return Polygon(zip(xs, ys))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def create_tile_grid_gdf(path: Union[Path, str], crs: str) -> gpd.GeoDataFrame:
    """Convert gdal_tile split info file into geopandas dataframe"""

    tiles_df = pd.read_csv(path, sep=";", header=None)
    tiles_df.columns = ["filename", "x1", "x2", "y1", "y2"]
    tiles_df["geometry"] = tiles_df.apply(make_poly, axis=1)
    tiles_df = tiles_df.drop(["x1", "x2", "y1", "y2"], axis=1)
    tiles_gpd = gpd.GeoDataFrame(tiles_df, crs=crs, geometry=tiles_df.geometry)
    return tiles_gpd


def _identify_empty(tile: Union[Path, str]) -> bool:
    """Helper func for exclude_nodata_tiles"""

    with xr.open_rasterio(tile).sel(band=1) as t:
        # original check
        # status = True if t.max().values - t.min().values > 0 else False
        # check 2 (edge tiles with all white/ black are also detected)
        return False if np.isin(t, [0, 255]).all() else True


def exclude_nodata_tiles(
    path: Iterable[Union[Path, str]],
    tiles_df: gpd.GeoDataFrame,
    workers: int,
) -> gpd.GeoDataFrame:
    """Identify tiles that only contain NoData (in parallel)"""

    print(f"WORKERS: {workers}")

    tile_names = sorted([Path(p) if isinstance(p, str) else p for p in path])

    results = process_map(_identify_empty, tile_names, max_workers=workers, chunksize=1)

    valid_d = dict([(t.name, r) for t, r in zip(tile_names, results)])

    tiles_df["status"] = tiles_df.filename.map(valid_d)
    # limit tiles to those with actual data (and delete status column afterwards)
    return tiles_df[tiles_df.status == 1].drop("status", axis=1)


# https://stackoverflow.com/a/39430508/5300574
def make_blocks_vectorized(x: np.ndarray, d: int) -> np.ndarray:
    """Discet an array into subtiles"""
    p, m, n = x.shape
    return (
        x.reshape(-1, m // d, d, n // d, d)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, p, d, d)
    )


def unmake_blocks_vectorized(x, d, m, n):
    """Merge subtiles back into array"""
    return (
        np.concatenate(x)
        .reshape(m // d, n // d, d, d)
        .transpose(0, 2, 1, 3)
        .reshape(m, n)
    )


def _split_tile(
    image: Path,
    *,
    source_dim: int,
    tile_size: int,
    format: str,
    outdir: Path,
) -> List[Tuple[str, bytes, bytes]]:
    """Helper func for split_tiles"""

    image_nir = Path(str(image.parent) + ".nir") / image.name.replace(
        "ESPG3044", "ESPG3044_NIR"
    )

    with rioxarray.open_rasterio(
        image, chunks={"band": 3, "x": tile_size, "y": tile_size}
    ) as t, rioxarray.open_rasterio(
        image_nir, chunks={"band": 1, "x": tile_size, "y": tile_size}
    ) as tnir:
        if len(t.x) * len(t.y) != source_dim ** 2:
            rgb_data = np.zeros((3, source_dim, source_dim), dtype=t.dtype)
            rgb_data[:, 0 : 0 + t.shape[1], 0 : 0 + t.shape[2]] = t.values

            nir_data = np.zeros((1, source_dim, source_dim), dtype=tnir.dtype)
            nir_data[:, 0 : 0 + tnir.shape[1], 0 : 0 + tnir.shape[2]] = tnir.values
        else:
            rgb_data = t.values
            nir_data = tnir.values

        mask_data = np.zeros((1, rgb_data.shape[1], rgb_data.shape[2]), dtype=np.uint8)

        subtile_rgb = make_blocks_vectorized(rgb_data, tile_size)
        subtile_nir = make_blocks_vectorized(nir_data, tile_size)
        subtile_mask = make_blocks_vectorized(mask_data, tile_size)

    samples = []
    if format == "TIFF":
        suffix = "tif"
    elif format == "PNG":
        suffix = "png"
    else:
        raise NotImplementedError

    for i in range(subtile_rgb.shape[0]):
        subtile_name = f"{image.name[:-4]}_{i:03d}"

        if np.min(subtile_rgb[i]) != np.max(subtile_rgb[i]):
            im = Image.fromarray(np.rollaxis(subtile_rgb[i], 0, 3))
            im_nir = Image.fromarray(subtile_nir[i].squeeze())
            im_mask = Image.fromarray(subtile_mask[i].squeeze())

            im_byte_arr = io.BytesIO()
            im.save(im_byte_arr, format=format)
            im_byte_arr = im_byte_arr.getvalue()

            im_nir_byte_arr = io.BytesIO()
            im_nir.save(im_nir_byte_arr, format=format)
            im_nir_byte_arr = im_nir_byte_arr.getvalue()

            im_mask_byte_arr = io.BytesIO()
            im_mask.save(im_mask_byte_arr, format=format)
            im_mask_byte_arr = im_mask_byte_arr.getvalue()

            sample = {
                "__key__": subtile_name,
                f"rgb.{suffix}": im_byte_arr,
                f"nir.{suffix}": im_nir_byte_arr,
                f"msk.{suffix}": im_mask_byte_arr,
                "txt": str(
                    round(
                        float(subtile_mask[i].sum()) / (tile_size * tile_size) * 100, 2
                    )
                ),
            }
            samples.append(sample)
    return samples


def split_tiles(images, workers: int, **kwargs) -> List[Any]:
    """Split tile into subtiles in parallel and save them to disk"""

    stats = []
    with wds.ShardWriter(
        str(Path(kwargs["outdir"]) / kwargs["shardpattern"]), maxcount=512
    ) as sink:

        del kwargs["shardpattern"]

        data = process_map(
            partial(_split_tile, **kwargs),
            images,
            max_workers=workers,
            chunksize=1,
        )

        for sample in reduce(lambda z, y: z + y, data):
            if sample:
                sink.write(sample)
                stats.append((sample["__key__"], sample["txt"]))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", type=Path)
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

    groundtruth_df = gpd.read_file(args.targets)
    tiles_df = create_tile_grid_gdf(
        args.input_dir / "locations.csv", groundtruth_df.crs
    )

    polygons, fileid, subtile_no = [], [], []

    for rcnt, row in tiles_df.iterrows():
        xmin, ymin, xmax, ymax = row.geometry.bounds

        xs = np.linspace(xmin, xmax, 17)
        ys = reversed(np.linspace(ymin, ymax, 17))

        subtile_cnt = 0
        for y1, y2 in list(pairwise(ys)):
            for x1, x2 in list(pairwise(xs)):
                polygons.append(Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)]))
                fileid.append(row.filename)
                subtile_no.append(subtile_cnt)
                subtile_cnt += 1

    subtile_grid = gpd.GeoDataFrame(
        {"filename": fileid, "subtile_id": subtile_no, "geometry": polygons},
        crs=groundtruth_df.crs,
    )

    merged_df = gpd.overlay(subtile_grid, groundtruth_df, how="intersection")
    dataset2_subtiles = [
        f"{Path(x).stem}_{y:03d}"
        for x, y in zip(merged_df.filename, merged_df.subtile_id)
    ]
    dataset2_subtiles = sorted(dataset2_subtiles)

    # ---- ds3
    stats = pd.read_csv("data/dataset/stats.csv")
    dataset3_subtiles = np.random.choice(
        stats[stats.deadtreepercent == 0.0].tile.values, 11 * 64
    )
    dataset3_subtiles = sorted(list(dataset3_subtiles))

    dataset2_tiles = list(set([x[:-4] for x in dataset2_subtiles]))
    dataset2_tile_paths = [args.input_dir / f"{x}.tif" for x in dataset2_tiles]

    # save to temporary directory
    cfg = dict(
        source_dim=8192,
        tile_size=512,
        outdir="data/extra",
        shardpattern="ds2-train-all-%06d.tar",
        format=suffix.upper(),
    )

    # we don't have masks, pass list of Nones
    workers = 8
    dataset2_subtile_stats = split_tiles(dataset2_tile_paths, workers, **cfg)  # noqa

    # ----------- REPLACE THIS WITH A SIMPLER SOURCE?

    tiles_df = exclude_nodata_tiles(
        sorted((Path("data") / "processed.images.2019").glob("*.tif")),
        tiles_df,
        workers,
    )

    # all possible tiles
    tile_files = list(tiles_df.filename.values)

    dataset1_tiles = [
        x.stem for x in sorted((Path("data") / "processed.masks.2019").glob("*.tif"))
    ]
    dataset1_tiles

    # all possible tiles - all non-deadtree tiles - all deadtree tiles
    dataset3_possible_tiles = list(
        set([x[:-4] for x in tile_files]) - set(dataset2_tiles) - set(dataset1_tiles)
    )
    len(dataset3_possible_tiles)

    dataset3_subtiles = (
        subtile_grid[
            subtile_grid.filename.isin([f"{x}.tif" for x in dataset3_possible_tiles])
        ]
        .sample(n=22 * 64, random_state=1234)
        .reset_index(drop=True)
    )

    dataset3_tiles = sorted(list(set(dataset3_subtiles.filename.values)))

    dataset3_tile_paths = [
        Path("data/processed.images.2019") / f"{x}" for x in dataset3_tiles
    ]

    cfg = dict(
        source_dim=8192,
        tile_size=512,
        outdir="data/extra",
        shardpattern="ds3-train-all-%06d.tar",
        format=suffix.upper(),
    )

    # we don't have masks, pass list of Nones
    workers = 8
    dataset3_subtile_stats = split_tiles(dataset3_tile_paths, workers, **cfg)  # noqa

    # --------------- output, extract samples from dataset2/ dataset3 datasets
    # iterate over all samples dataset2

    dataset = wds.WebDataset("data/extra/ds2-train-all-{000000..000016}.tar")
    cnt = 0

    with wds.ShardWriter("data/extra/ds2-train-%06d.tar", maxcount=64) as sink:
        for sample in dataset:
            if sample["__key__"] in dataset2_subtiles:
                # write out
                sink.write(sample)
                cnt += 1

    dataset3_subtilesB = [
        f"{Path(x).stem}_{y:03d}"
        for x, y in zip(dataset3_subtiles.filename, dataset3_subtiles.subtile_id)
    ]
    dataset3_subtilesB = sorted(dataset3_subtilesB)

    # iterate over all samples dataset3

    dataset = wds.WebDataset("data/extra/ds3-train-all-{000000..000353}.tar")
    cnt = 0

    with wds.ShardWriter("data/extra/ds3-train-%06d.tar", maxcount=64) as sink:
        for sample in dataset:
            if sample["__key__"] in dataset3_subtilesB:
                # write out
                sink.write(sample)
                cnt += 1


if __name__ == "__main__":
    main()
