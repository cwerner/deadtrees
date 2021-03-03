import os
from functools import partial
from pathlib import Path
from typing import Iterable, List, TypeVar, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# CONSTANTS
WORKERS = 8
TILE_SIZE = 512
SRC_PATH = Path.cwd() / "data"


def make_poly(coords: pd.Series) -> Polygon:
    """Create Shapely polygon (a tile boundary) from x1,y1 and x2,y2 coordinates"""
    xs = [coords[v] for v in "x1,x1,x2,x2,x1".split(",")]
    ys = [coords[v] for v in "y1,y2,y2,y1,y1".split(",")]
    return Polygon(zip(xs, ys))


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


def _split_tile(tile_name: Path, tile_mask_name: Path) -> List[float]:
    """Helper func for split_tiles"""

    ratios = []
    with rioxarray.open_rasterio(
        tile_name, chunks={"band": 3, "x": TILE_SIZE, "y": TILE_SIZE}
    ) as t, rioxarray.open_rasterio(
        tile_mask_name, chunks={"band": 1, "x": TILE_SIZE, "y": TILE_SIZE}
    ) as tm:
        subtile_rgb = make_blocks_vectorized(t.values, TILE_SIZE)
        subtile_mask = make_blocks_vectorized(tm.values, TILE_SIZE)

    for i in range(subtile_rgb.shape[0]):
        subtile_name = f"{tile_mask_name.name[:-4]}_{i:03d}.tif"

        im = Image.fromarray(np.rollaxis(subtile_rgb[i], 0, 3))
        im.save(SRC_PATH / "dataset" / "train" / "image" / subtile_name)

        im_mask = Image.fromarray(subtile_mask[i].squeeze())
        im_mask.save(SRC_PATH / "dataset" / "train" / "mask" / subtile_name)

        ratios.append(
            round(float(subtile_mask[i].sum()) / (TILE_SIZE * TILE_SIZE) * 100, 2)
        )

    return ratios


def split_tiles(tiles, tile_masks, workers: int = WORKERS) -> List[List[float]]:
    """Split tile into subtiles in parallel and save them to disc"""

    return process_map(_split_tile, tiles, tile_masks, max_workers=workers, chunksize=1)


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
        status = True if t.max().values - t.min().values > 0 else False
    return status


def exclude_nodata_tiles(
    path: Iterable[Union[Path, str]], tiles_df: gpd.GeoDataFrame, workers: int = WORKERS
) -> gpd.GeoDataFrame:
    """Identify tiles that only contain NoData (in parallel)"""

    tile_names = sorted([Path(p) if isinstance(p, str) else p for p in path])

    results = process_map(_identify_empty, tile_names, max_workers=workers, chunksize=1)

    valid_d = dict([(t.name, r) for t, r in zip(tile_names, results)])
    tiles_df["status"] = tiles_df.filename.map(valid_d)

    # limit tiles to those with actual data (and delete status column afterwards)
    return tiles_df[tiles_df.status == 1].drop("status", axis=1)


def split_groundtruth_data_by_tiles(
    dtree: gpd.GeoDataFrame, tiles_df: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Split the oberserved dead tree areas into tile segments for fast er downstream processing"""

    union_gpd = gpd.overlay(dtree, tiles_df, how="union")
    union_gpd["id"] = union_gpd.id.fillna(1)
    union_gpd.loc[union_gpd.id > 1, "id"] = 2

    train_files = list(
        sorted(union_gpd[union_gpd.id == 2].filename.value_counts().keys())
    )

    tiles_with_groundtruth = tiles_df[tiles_df.filename.isin(train_files)]
    return tiles_with_groundtruth


def _mask_tile(
    tile_filename: str,
    *,
    groundtruth_df: gpd.GeoDataFrame,
    crs: str,
    base_path: Path,
) -> float:

    image_tile_path = base_path / "tiles" / tile_filename
    mask_tile_path = base_path / "tiles_mask" / tile_filename

    with rioxarray.open_rasterio(
        image_tile_path, chunks={"band": 3, "x": 512, "y": 512}
    ) as tile:
        mask = xr.ones_like(tile.load().sel(band=1, drop=True), dtype="uint8")

        mask.rio.set_crs(crs)
        selection = groundtruth_df[groundtruth_df.filename == tile_filename]
        mask = mask.rio.clip(
            selection.geometry,
            crs,
            drop=False,
            invert=True,
            all_touched=True,
            from_disk=True,
        )
        mask.rio.to_raster(mask_tile_path, tiled=True)
        mask_sum = mask.sum()  # just for checks
    return mask_sum


def create_tile_mask_geotiffs(
    tiles_df_train: gpd.GeoDataFrame, workers: int = WORKERS, **kwargs
) -> None:
    """Create binary mask geotiffs"""
    _ = process_map(
        partial(_mask_tile, **kwargs),
        tiles_df_train.filename.values,
        max_workers=workers,
        chunksize=1,
    )


# PROCESSIGN STAGES
def stage_0() -> None:
    """
    Preliminary processing stage: go from original geotiff to 8192x8192px tiles (using GDAL)
    """
    target_dir = "data/tiles"
    src_file = "/glacier2/dldatasets/deadtrees/tif_ortho_2019.tif"
    summary_file = "locations.csv"
    size = 8192

    cmd = f"gdal_retile.py -csv {summary_file} -v -ps {size} {size} -targetDir {target_dir} {src_file}"
    os.system(cmd)


def stage_1() -> None:
    """
    Stage 1: produce masks for training tiles
    """

    # load domain shape files and use its crs for the entire script
    dtree = gpd.read_file(SRC_PATH / "shp" / "dead_treed_proj.shp")
    crs = dtree.crs  # reference crs

    tiles_df = create_tile_grid_gdf(SRC_PATH / "tiles" / "locations.csv", crs)
    tiles_df = exclude_nodata_tiles((SRC_PATH / "tiles").glob("*.tif"), tiles_df)

    tiles_df_train = split_groundtruth_data_by_tiles(dtree, tiles_df)

    create_tile_mask_geotiffs(
        tiles_df_train, groundtruth_df=tiles_df_train, crs=crs, base_path=SRC_PATH
    )


def stage_2() -> None:
    """
    Stage 2: produce actual model dataset (currently file based/ later hopefully as hdf5 file)
    """

    # subtile_stats = split_tiles(train_files)
    train_mask_files = sorted((SRC_PATH / "tiles_mask").glob("*.tif"))

    def fix_paths(in_path):
        return Path(str(in_path).replace("tiles_mask", "tiles"))

    train_files = [fix_paths(x) for x in train_mask_files]

    subtile_stats = split_tiles(train_files, train_mask_files)

    with open(Path.cwd() / "subtile_stats.csv", "w") as fout:
        fout.write("tile,subtile,subtile_id,deadtreepercent\n")
        for i, fname in enumerate(train_files):
            for j, stats in enumerate(subtile_stats[i]):
                line = f"{fname.name},{fname.name[:-4]}_{j:03d}.tif,{j},{stats}\n"
                fout.write(line)


def main():
    # stage_0()  # disabled, run only once
    stage_1()
    # stage_2()


if __name__ == "__main__":
    main()
