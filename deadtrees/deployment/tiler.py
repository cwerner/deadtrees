import argparse
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rioxarray
import torch
from deadtrees.data.deadtreedata import val_transform
from deadtrees.deployment.inference import ONNXInference, PyTorchInference
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def is_tilable(infile: Union[str, Path]):
    with rioxarray.open_rasterio(infile) as da:
        assert da.sel(band=1, drop=True).shape == (
            8192,
            8192,
        ), "expected a geotiff with spatial size 8192x8192px"
    return True


class Tiler:
    def __init__(self, infile: Optional[Union[str, Path]] = None) -> None:
        self._infile = infile
        self._source = None
        self._target = None

    def load_file(self, infile: Union[str, Path]) -> None:
        if is_tilable(infile):
            self._infile = infile
            self._source = rioxarray.open_rasterio(
                self._infile, chunks={"band": 3, "x": 512, "y": 512}
            )
            self._target = (
                self._source.sel(band=1, drop=True).astype("uint8").copy(deep=True)
            )

    def write_file(self, outfile: Union[str, Path]) -> None:
        if self._target is not None:
            self._target.rio.to_raster(outfile, compress="LZW", tiled=True)

    def get_batches(self):
        return make_blocks_vectorized(self._source.values, 512)

    def pass_batches(self, batches: np.ndarray):
        self._target = self._target.load()
        self._target.loc[:] = unmake_blocks_vectorized(batches, 512, 8192, 8192)


# https://stackoverflow.com/a/39430508/5300574
def make_blocks_vectorized(x: np.ndarray, d: int) -> np.ndarray:
    """Discet an array into subtiles"""
    p, m, n = x.shape
    return (
        x.reshape(-1, m // d, d, n // d, d)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, p, d, d)
    )


def unmake_blocks_vectorized(x: np.ndarray, d: int, m: int, n: int) -> np.ndarray:
    """Merge subtiles back into array"""
    return (
        np.concatenate(x)
        .reshape(m // d, n // d, d, d)
        .transpose(0, 2, 1, 3)
        .reshape(m, n)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument(
        "-o",
        dest="outpath",
        type=Path,
        default=Path("."),
        help="output directory",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        dest="all",
        default=False,
        help="process complete directory",
    )

    args = parser.parse_args()

    bs = 8

    INFILE = args.infile

    def is_valid_tile(infile):
        with rioxarray.open_rasterio(infile).sel(band=1) as t:
            return False if np.isin(t, [0, 255]).all() else True

    # inference = ONNXInference("checkpoints/bestmodel.onnx")
    inference = PyTorchInference("checkpoints/bestmodel.ckpt")

    #
    # model.to(device)

    if args.all:
        INFILES = sorted(INFILE.glob("ortho*.tif"))
    else:
        INFILES = [INFILE]

    for INFILE in INFILES:

        if not is_valid_tile(INFILE):
            continue

        tiler = Tiler()
        tiler.load_file(INFILE)
        batches = np.split(tiler.get_batches(), 256 // bs, axis=0)
        out_batches = []

        for b, batch in enumerate(tqdm(batches, desc=INFILE.name)):
            batch_tensor = torch.stack(
                [val_transform(image=i.transpose(1, 2, 0))["image"] for i in batch]
            )

            # onnx
            # out_batch = inference.run(batch_tensor.detach().cpu().numpy())

            # pytorch
            out_batch = (
                inference.run(batch_tensor.detach().to("cuda"), device="cuda")
                .cpu()
                .numpy()
            )

            out_batches.append(out_batch)

        OUTFILE = args.outpath / INFILE.name
        OUTFILE_PREVIEW = Path(str(args.outpath) + "_preview") / INFILE.name

        tiler.pass_batches(np.concatenate(out_batches, axis=0))
        tiler.write_file(OUTFILE)

        #
        from PIL import Image

        image = Image.fromarray(np.uint8(tiler._target.values * 255), "L")
        image.save(OUTFILE_PREVIEW)


if __name__ == "__main__":
    main()
