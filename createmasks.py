# createmasks stage

import argparse
from ast import Num
from functools import partial
from pathlib import Path
from typing import Iterable, List, TypeVar, Union

import psutil

import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def make_poly(coords: pd.Series) -> Polygon:
    """Create Shapely polygon (a tile boundary) from x1,y1 and x2,y2 coordinates"""
    xs = [coords[v] for v in "x1,x1,x2,x2,x1".split(",")]
    ys = [coords[v] for v in "y1,y2,y2,y1,y1".split(",")]
    return Polygon(zip(xs, ys))


def _identify_empty(tile: Union[Path, str]) -> bool:
    """Helper func for exclude_nodata_tiles"""

    with xr.open_rasterio(tile).sel(band=1) as t:
        status = True if t.max().values - t.min().values > 0 else False
    return status


def exclude_nodata_tiles(
    path: Iterable[Union[Path, str]], tiles_df: gpd.GeoDataFrame, workers: int
) -> gpd.GeoDataFrame:
    """Identify tiles that only contain NoData (in parallel)"""

    print(f"WORKERS: {workers}")

    tile_names = sorted([Path(p) if isinstance(p, str) else p for p in path])

    results = process_map(_identify_empty, tile_names, max_workers=workers, chunksize=1)

    valid_d = dict([(t.name, r) for t, r in zip(tile_names, results)])

    tiles_df["status"] = tiles_df.filename.map(valid_d)
    tiles_df.to_file("exclude.shp")
    # limit tiles to those with actual data (and delete status column afterwards)
    return tiles_df[tiles_df.status == 1].drop("status", axis=1)


# ok
def create_tile_grid_gdf(path: Union[Path, str], crs: str) -> gpd.GeoDataFrame:
    """Convert gdal_tile split info file into geopandas dataframe"""

    tiles_df = pd.read_csv(path, sep=";", header=None)
    tiles_df.columns = ["filename", "x1", "x2", "y1", "y2"]
    tiles_df["geometry"] = tiles_df.apply(make_poly, axis=1)
    tiles_df = tiles_df.drop(["x1", "x2", "y1", "y2"], axis=1)
    tiles_gpd = gpd.GeoDataFrame(tiles_df, crs=crs, geometry=tiles_df.geometry)
    return tiles_gpd


def split_groundtruth_data_by_tiles(
    dtree: gpd.GeoDataFrame, tiles_df: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Split the oberserved dead tree areas into tile segments for faster downstream processing"""

    union_gpd = gpd.overlay(dtree, tiles_df, how="union")
    union_gpd["id"] = union_gpd.id.fillna(1)
    union_gpd.loc[union_gpd.id > 1, "id"] = 2

    union_gpd.to_file("union.shp")

    train_files = list(
        sorted(union_gpd[union_gpd.id == 2].filename.value_counts().keys())
    )

    tiles_with_groundtruth = tiles_df[tiles_df.filename.isin(train_files)]
    return tiles_with_groundtruth, union_gpd[union_gpd.id == 2]


def _mask_tile(
    tile_filename: str,
    *,
    groundtruth_df: gpd.GeoDataFrame,
    crs: str,
    inpath: Path,
    outpath: Path,
) -> float:

    image_tile_path = inpath / tile_filename
    mask_tile_path = outpath / tile_filename

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
            invert=False,
            all_touched=True,
            from_disk=True,
        )
        mask.rio.to_raster(mask_tile_path, tiled=True)
        mask_sum = float(mask.sum().values)  # just for checks
    return mask_sum


def create_tile_mask_geotiffs(
    tiles_df_train: gpd.GeoDataFrame, workers: int, **kwargs
) -> None:
    """Create binary mask geotiffs"""
    pct = process_map(
        partial(_mask_tile, **kwargs),
        tiles_df_train.filename.values,
        max_workers=workers,
        chunksize=1,
    )
    print(pct)


def create_masks(indir: str, outdir: str, shpfile: str, workers: int) -> None:
    """
    Stage 1: produce masks for training tiles
    """

    IN_PATH = Path(indir)
    OUT_PATH = Path(outdir)

    # load domain shape files and use its crs for the entire script
    dtree = gpd.read_file(shpfile)
    crs = dtree.crs  # reference crs

    tiles_df = create_tile_grid_gdf(IN_PATH / "locations.csv", crs)
    tiles_df = exclude_nodata_tiles(sorted(IN_PATH.glob("*.tif")), tiles_df, workers)
    print(f"len2: {len(tiles_df)}")

    tiles_df_train, groundtruth_df = split_groundtruth_data_by_tiles(dtree, tiles_df)
    print(f"len3: {len(tiles_df_train)}")

    create_tile_mask_geotiffs(
        tiles_df_train,
        workers,
        groundtruth_df=groundtruth_df,
        crs=crs,
        inpath=IN_PATH,
        outpath=OUT_PATH,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outdir")
    parser.add_argument("shpfile")

    num_cores = psutil.cpu_count(logical=False)

    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=num_cores,
        help="number of workers for parallel execution [def: %(default)s]",
    )

    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    create_masks(args.indir, args.outdir, args.shpfile, args.workers)


if __name__ == "__main__":
    main()