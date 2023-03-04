import argparse
import math
from pathlib import Path
from typing import List

from tiler import Merger, Tiler

import numpy as np
import rioxarray
import torch
import xarray as xr
from deadtrees.data.deadtreedata import val_transform
from deadtrees.deployment.inference import PyTorchEnsembleInference, PyTorchInference
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)

    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        action="append",
        type=Path,
        default=[],
        help="model artefact",
    )

    parser.add_argument(
        "-o",
        dest="outpath",
        type=Path,
        default=Path("."),
        help="output directory",
    )

    parser.add_argument(
        "--overlap",
        dest="overlap",
        type=int,
        default=32,
        help="overlap of subtiles (256x256px tile: 32 or 128, def:32)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        dest="all",
        default=False,
        help="process complete directory",
    )

    args = parser.parse_args()

    if len(args.model) == 0:
        args.model = [Path("checkpoints/bestmodel.ckpt")]

    if args.overlap not in [32, 128]:
        print("Currently only 32 or 128 allowed for subtile overlap")
        exit(-1)

    bs = 64

    INFILE = args.infile

    def is_valid_tile(infile):
        with rioxarray.open_rasterio(infile).sel(band=1) as t:
            return False if np.isin(t, [0, 255]).all() else True

    if len(args.model) == 1:
        print("Default inference: single model")
        inference = PyTorchInference(args.model[0])
    else:
        print(f"Ensemble inference: {len(args.model)} models")
        inference = PyTorchEnsembleInference(*args.model)

    n_channel = inference.channels
    n_classes = inference.classes

    print(
        f"Inference using {n_channel}-channel model(s) and {n_classes} output classes"
    )

    if args.all:
        INFILES = sorted(INFILE.glob("ortho*.tif"))
    else:
        INFILES = [INFILE]

    for INFILE in INFILES:

        if not is_valid_tile(INFILE):
            continue

        OUTFILE = args.outpath / INFILE.name

        # read geotiff
        source: xr.DataArray = rioxarray.open_rasterio(
            INFILE, chunks={"band": 4, "x": 256, "y": 256}
        )
        if n_channel > len(source.band):
            print(
                f"Source image {source.name} has wrong number of channels: image: {len(source.band)} model: {n_channel}"
            )

        # prepare output geotiff
        target: xr.DataArray = (
            source.sel(band=1, drop=True).astype("uint8").copy(deep=True).load()
        )

        in_tiler = Tiler(
            data_shape=source.values.shape,
            tile_shape=(n_channel, 256, 256),
            overlap=args.overlap,
            channel_dimension=0,
        )
        out_tiler = Tiler(
            data_shape=source.values.shape,
            tile_shape=(n_classes, 256, 256),
            overlap=args.overlap,
            channel_dimension=0,
        )

        # one merger for each model
        out_merger = [Merger(out_tiler)] * len(args.model)

        # make sure n_channels of model match data size (use RGB aka 0:3 if model requires)
        batches = [
            batch
            for _, batch in in_tiler(source.values[0:n_channel, ...], batch_size=bs)
        ]

        for batch_id, batch in enumerate(tqdm(batches, desc=INFILE.name)):
            batch_tensor = torch.stack(
                [val_transform(image=i.transpose(1, 2, 0))["image"] for i in batch]
            )

            out_batch = (
                inference.run(
                    batch_tensor.detach().to("cuda"), device="cuda", return_raw=True
                )
                .cpu()
                .numpy()
            )

            # dims: model, bs, c, h, w
            if isinstance(inference, PyTorchEnsembleInference):
                for i in range(len(args.model)):
                    out_merger[i].add_batch(batch_id, bs, out_batch[i])
            else:
                out_merger[0].add_batch(batch_id, bs, out_batch)

        # this is still based on logits since we used return_raw in inference.run() !
        # 1) use merger to recreate full tile
        # 2) argmax over logits to find dominent class in each pixel
        # 3) take the mode over all models to derive final px class value (this is done via torch)
        output_per_model = np.array(
            [np.argmax(m.merge(unpad=True), axis=0) for m in out_merger]
        )
        output = torch.mode(torch.Tensor(output_per_model), axis=0).values.numpy()

        target[:] = output
        target.rio.to_raster(OUTFILE, compress="LZW", tiled=True)


if __name__ == "__main__":
    main()
