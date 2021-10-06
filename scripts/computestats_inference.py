import argparse
from functools import partial, reduce
from pathlib import Path

from joblib import delayed, Parallel

import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm

classes = [0, 1, 2]
WORKERS = 16


def process_tile(tile, *, year):
    with rioxarray.open_rasterio(tile, chunks=(1, 512, 512)).squeeze(drop=True) as ds:
        # TODO: we only consider deadtrees (not potential classes of them)
        # ds = ds.clip(max=1)

        unique, counts = np.unique(ds.values, return_counts=True)
    row_data = dict(zip([f"cl_{int(x)}" for x in unique], counts))

    for c in classes:
        if f"cl_{c}" not in row_data:
            row_data[f"cl_{c}"] = 0

    row_data["total"] = int(ds.count().compute())
    row_data["tile"] = tile.stem.replace(f"ortho_ms_{year}_EPSG3044_", "")
    return row_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=Path, nargs="+")

    args = parser.parse_args()

    dfs = []
    years = [2017, 2018, 2019, 2020]
    for year in years:
        inpath = None
        for dpath in args.datapath:
            if f"predicted.{year}" in str(dpath):
                inpath = dpath
        if not inpath:
            raise NotImplementedError

        print(f"Processing year: {year}...")
        tiles = sorted(inpath.glob("*.tif"))

        results = Parallel(n_jobs=WORKERS)(
            delayed(partial(process_tile, year=year))(x) for x in tqdm(tiles)
        )

        df = pd.DataFrame(results)
        df["deadarea_m2"] = (
            (df["cl_1"] + df["cl_2"]) * 0.200022269188281 * 0.200022454940277
        ).round(1)
        dfs.append(df)

    # add suffixes to each df (but remove if again form the join column <tile>)
    dfs = [df.add_suffix(f"_{s}") for df, s in zip(dfs, years)]
    dfs = [df.rename(columns={f"tile_{s}": "tile"}) for df, s in zip(dfs, years)]

    dfall = reduce(lambda x, y: pd.merge(x, y, on=["tile"], how="outer"), dfs)

    # cleanup, drop all but one columns total and move tile, total cols to front
    dfall = dfall.rename(columns={f"total_{years[0]}": "total"})
    dfall = dfall[dfall.columns.drop(list(dfall.filter(regex="total_")))]

    colnames = list(dfall)
    colnames.insert(0, colnames.pop(colnames.index("total")))
    colnames.insert(0, colnames.pop(colnames.index("tile")))
    dfall = dfall.loc[:, colnames].convert_dtypes()
    dfall.to_csv(args.datapath[0].parent / "predicted.stats.csv", index=False)


# merge

if __name__ == "__main__":
    main()
