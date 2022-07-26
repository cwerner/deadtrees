import argparse
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Optional, Tuple

from joblib import delayed, Parallel

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from shapely.geometry import box, Polygon
from tqdm import tqdm

classes = [0, 1, 2]
WORKERS = 16


lat_point_list = [50.854457, 52.518172, 50.072651, 48.853033, 50.854457]
lon_point_list = [4.377184, 13.407759, 14.435935, 2.349553, 4.377184]

polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
crs = {"init": "epsg:4326"}


@dataclass
class Result:
    bounds: Tuple[float, float, float, float]
    crs: Any
    conifer: Optional[float]
    broadleaf: Optional[float]

    @property
    def total(self) -> Optional[float]:
        if (self.conifer is None) and (self.broadleaf is None):
            return None
        if (self.conifer is None) and (self.broadleaf is not None):
            raise NotImplementedError
        if (self.conifer is not None) and (self.broadleaf is None):
            raise NotImplementedError
        return self.conifer + self.broadleaf

    def to_gdf(self):
        if self.bounds and self.crs:
            gdf = gpd.GeoDataFrame(
                index=[0],
                data={
                    "conifer": [self.conifer],
                    "broadleaf": [self.broadleaf],
                    "total": [self.total],
                },
                crs=self.crs,
                geometry=[box(*self.bounds)],
            )
            return gdf
        return None


def process_tile(tile: Path, forest_tile: Path, *, year: str, limit: int) -> Result:
    with rioxarray.open_rasterio(tile, chunks=(1, 512, 512)).squeeze(
        drop=True
    ) as ds, rioxarray.open_rasterio(forest_tile, chunks=(1, 512, 512)).squeeze(
        drop=True
    ) as ds_mask:

        res = []
        for c in classes[1:]:
            a = ds.values
            b = ds_mask.values

            if (b.sum() / b.size) * 100 < limit:
                return Result(
                    conifer=None, broadleaf=None, bounds=ds.rio.bounds(), crs=ds.rio.crs
                )
            dead = a[(a == c) & (b == 1)].sum()
            forest = b.sum()
            res.append((dead / forest) * 100)
        return Result(
            conifer=res[0], broadleaf=res[1], bounds=ds.rio.bounds(), crs=ds.rio.crs
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        default=10,
        help="Min. forest cover to include pixel [%]",
    )
    parser.add_argument("datapath", type=Path, nargs="+")

    args = parser.parse_args()

    years = [2017, 2018, 2019, 2020]
    for year in years:
        inpath = None
        for dpath in args.datapath:
            if f"processed.lus.{year}" in str(dpath):
                inpath = dpath
        if not inpath:
            raise NotImplementedError

        print(f"Processing year: {year}...")
        tiles_forest_mask = sorted(inpath.glob("*.tif"))

        def swap_dir(x: Path, search: str, replace: str) -> Path:
            path_elements = list(x.parts)
            idx = path_elements.index(search)
            path_elements[idx] = replace
            return Path(*path_elements)

        tiles = [
            swap_dir(t, f"processed.lus.{year}", f"predicted.{year}")
            for t in tiles_forest_mask
        ]

        results = Parallel(n_jobs=WORKERS)(
            delayed(partial(process_tile, year=year, limit=args.limit))(*d)
            for d in tqdm(list(zip(tiles, tiles_forest_mask)))
        )

        gpd.GeoDataFrame(
            pd.concat([r.to_gdf() for r in results], ignore_index=True)
        ).to_file(f"data/aggregated_{year}.shp")


if __name__ == "__main__":
    main()
