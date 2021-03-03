import pandas as pd 
import geopandas as gpd
from pathlib import Path

import xarray as xr
import rioxarray

import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from functools import partial

from shapely.geometry import Polygon

from typing import Union

from PIL import Image


import os

def make_poly(coords):
    xs = [coords['x1'], coords['x1'], coords['x2'], coords['x2'], coords['x1']]
    ys = [coords['y1'], coords['y2'], coords['y2'], coords['y1'], coords['y1']]
    return Polygon(zip(xs, ys))

# Type aliases
GeoDataFrame = gpd.GeoDataFrame


# CONSTANTS
WORKERS = 8
TILE_SIZE = 512
SRC_PATH = Path.cwd() / "data"


# https://stackoverflow.com/a/39430508/5300574
def make_blocks_vectorized(x,d):
    p,m,n = x.shape
    return x.reshape(-1,m//d,d,n//d,d).transpose(1,3,0,2,4).reshape(-1,p,d,d)

def unmake_blocks_vectorized(x,d,m,n):    
    return np.concatenate(x).reshape(m//d,n//d,d,d).transpose(0,2,1,3).reshape(m,n)

def _split_tile(tile_name: Path, tile_mask_name: Path):

    ratios = []
    with rioxarray.open_rasterio(tile_name, chunks={'band': 3, 'x': TILE_SIZE, 'y': TILE_SIZE}) as t, \
         rioxarray.open_rasterio(tile_mask_name, chunks={'band': 1, 'x': TILE_SIZE, 'y': TILE_SIZE}) as tm:
        subtile_rgb = make_blocks_vectorized(t.values, TILE_SIZE)
        subtile_mask = make_blocks_vectorized(tm.values, TILE_SIZE)

    for i in range(subtile_rgb.shape[0]):
        im_path = SRC_PATH / "dataset" / "train" / "image" / f"{tile_name.name[:-4]}_{i:03d}.tif"
        im = Image.fromarray(np.rollaxis(subtile_rgb[i], 0,3))
        im.save(im_path)

        im_mask_path = SRC_PATH / "dataset" / "train" / "mask" / f"{tile_mask_name.name[:-4]}_{i:03d}.tif"
        im_mask = Image.fromarray(subtile_mask[i].squeeze())
        im_mask.save(im_mask_path)

        ratios.append(round(float(subtile_mask[i].sum()) / (TILE_SIZE*TILE_SIZE)*100, 2))

    return ratios

def split_tiles(tiles, tile_masks, max_workers=WORKERS):
    stats = process_map(_split_tile, tiles, tile_masks, max_workers=max_workers, chunksize=1)
    return stats


def create_tile_grid_gdf(path: Union[Path, str], crs:str) -> GeoDataFrame:
    """Convert gdal_tile split info file into geopandas dataframe"""
    tiles_df = pd.read_csv(path, sep=";", header=None)
    tiles_df.columns = ['filename', 'x1', 'x2', 'y1', 'y2']
    tiles_df['geometry'] = tiles_df.apply(make_poly, axis=1)
    tiles_df = tiles_df.drop(['x1','x2','y1','y2'], axis=1)
    tiles_gpd = gpd.GeoDataFrame(tiles_df, crs=crs, geometry=tiles_df.geometry)
    return tiles_gpd


def _identify_empty(tile):
    with xr.open_rasterio(tile).sel(band=1) as t:
        status = 1 if t.max().values - t.min().values > 0 else 0
    return status

def exclude_nodata_tiles(path: Union[Path, str], tiles_df: GeoDataFrame, workers=WORKERS) -> GeoDataFrame:
    """Identify tiles that only contain NoData (in parallel)"""
    tile_names = sorted(path)
    
    results =  process_map(_identify_empty, tile_names, max_workers=workers, chunksize=1)

    valid_d = dict([(t.name, r) for t, r in zip(tile_names, results)])
    tiles_df["status"] = tiles_df.filename.map(valid_d)

    # limit tiles to those with actual data (and delete status column afterwards)
    tiles_df = tiles_df[tiles_df.status == 1]
    del tiles_df["status"]
    return tiles_df

def split_groundtruth_data_by_tiles(dtree: GeoDataFrame, tiles_df: GeoDataFrame):
    # intersect dtree
    union_gpd = gpd.overlay(dtree, tiles_df, how='union')
    union_gpd['id'] = union_gpd.id.fillna(1)
    union_gpd.loc[union_gpd.id > 1, 'id'] = 2

    train_files = list(sorted(union_gpd[union_gpd.id == 2].filename.value_counts().keys()))

    tiles_with_groundtruth = tiles_df[tiles_df.filename.isin(train_files)]
    return tiles_with_groundtruth


def _mask_tile(tile_filename: str, groundtruth_df: GeoDataFrame=None, crs=None, base_path=None):

    image_tile_path = base_path / "tiles" / tile_filename
    mask_tile_path = base_path / "tiles_mask.dummy" / tile_filename
    
    with rioxarray.open_rasterio(image_tile_path, chunks={'band': 3, 'x': 512, 'y': 512}) as tile:
        mask = xr.ones_like(tile.load().sel(band=1, drop=True), dtype='uint8')

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
        mask_sum = mask.sum()   # just for checks
    return mask_sum


def create_tile_mask_geotiffs(tiles_df_train, workers=WORKERS, **kwargs) -> None:  
    _ = process_map(partial(_mask_tile, **kwargs), tiles_df_train.filename.values, max_workers=workers, chunksize=1)



def stage_0():
    """
    Preliminary processing stage: go from original geotiff to 8192x8192px tiles (using GDAL)
    """
    target_dir = 'data/tiles'
    src_file = '/glacier2/dldatasets/deadtrees/tif_ortho_2019.tif'
    summary_file = 'locations.csv'
    size = 8192

    cmd = f"gdal_retile.py -csv {summary_file} -v -ps {size} {size} -targetDir {target_dir} {src_file}"
    os.system(cmd)

def stage_1():
    """
    Stage 1: produce masks for training tiles
    """

    # load domain shape files and use its crs for the entire script
    dtree = gpd.read_file(SRC_PATH / "shp" / "dead_treed_proj.shp")
    crs = dtree.crs    # reference crs

    tiles_df = create_tile_grid_gdf(SRC_PATH / "tiles" / "locations.csv", crs)
    tiles_df = exclude_nodata_tiles((SRC_PATH / "tiles").glob('*.tif'), tiles_df)

    tiles_df_train = split_groundtruth_data_by_tiles(dtree, tiles_df)

    create_tile_mask_geotiffs(tiles_df_train, groundtruth_df=tiles_df_train, crs=crs, base_path=SRC_PATH)


def stage_2():
    """
    Stage 2: produce actual model dataset (currently file based/ later heopefully as hdf5 file)
    """

    TGT_PATH = SRC_PATH / "dataset" / "train"   # path where images and masks with tile_size will go

    #subtile_stats = split_tiles(train_files)
    train_mask_files = sorted((SRC_PATH / "tiles_mask").glob("*.tif"))

    def fix_paths(in_path):
        return Path(str(in_path).replace("tiles_mask", "tiles"))
    
    train_files = [fix_paths(x) for x in train_mask_files]

    subtile_stats = split_tiles(train_files, train_mask_files)

    with open(Path.cwd() / "subtile_stats.csv", 'w') as fout:
        fout.write("tile,subtile,subtile_id,deadtreepercent\n")
        for i, fname in enumerate(train_files):
            for j, stats in enumerate(subtile_stats[i]):
                line = f"{fname.name},{fname.name[:-4]}_{j:03d}.tif,{j},{stats}\n"
                fout.write(line)



def main():
    # stage_0()  # disabled, run only once
    #stage_1()
    stage_2()


if __name__ == "__main__":
    main()