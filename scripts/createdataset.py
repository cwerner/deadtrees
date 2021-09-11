import argparse
import io
from functools import partial, reduce
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, List, Tuple, TypeVar, Union

import psutil
import webdataset as wds

import numpy as np
import rioxarray
import xarray as xr
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


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
    mask: Path,
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
    ) as tnir, rioxarray.open_rasterio(
        mask, chunks={"band": 1, "x": tile_size, "y": tile_size}
    ) as tm:
        if len(t.x) * len(t.y) != source_dim ** 2:
            rgb_data = np.zeros((3, source_dim, source_dim), dtype=t.dtype)
            rgb_data[:, 0 : 0 + t.shape[1], 0 : 0 + t.shape[2]] = t.values

            nir_data = np.zeros((1, source_dim, source_dim), dtype=tnir.dtype)
            nir_data[:, 0 : 0 + tnir.shape[1], 0 : 0 + tnir.shape[2]] = tnir.values

            mask_data = np.zeros((1, source_dim, source_dim), dtype=tm.dtype)
            mask_data[:, 0 : 0 + tm.shape[1], 0 : 0 + tm.shape[2]] = tm.values
        else:
            rgb_data = t.values
            nir_data = tnir.values
            mask_data = tm.values
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
        subtile_name = f"{mask.name[:-4]}_{i:03d}"

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


def split_tiles(images, masks, workers: int, **kwargs) -> List[Any]:
    """Split tile into subtiles in parallel and save them to disk"""

    stats = []
    with wds.ShardWriter(
        str(kwargs["outdir"]) + "/train/train-%06d.tar", maxcount=512
    ) as sink:

        data = process_map(
            partial(_split_tile, **kwargs),
            images,
            masks,
            max_workers=workers,
            chunksize=1,
        )

        for sample in reduce(lambda z, y: z + y, data):
            if sample:
                sink.write(sample)
                stats.append((sample["__key__"], sample["txt"]))

    return stats


def _split_inference_tile(
    image: Path,
    *,
    source_dim: int,
    tile_size: int,
    format: str,
    outdir: Path,
) -> List[float]:
    """Helper func for split_inference_tiles"""

    with rioxarray.open_rasterio(
        image, chunks={"band": 3, "x": tile_size, "y": tile_size}
    ) as t:
        if len(t.x) * len(t.y) != source_dim ** 2:
            rgb_data = np.zeros((3, source_dim, source_dim), dtype=t.dtype)
            rgb_data[:, 0 : 0 + t.shape[1], 0 : 0 + t.shape[2]] = t.values
        else:
            rgb_data = t.values
        subtile_rgb = make_blocks_vectorized(rgb_data, tile_size)

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

            im_byte_arr = io.BytesIO()
            im.save(im_byte_arr, format=format)
            im_byte_arr = im_byte_arr.getvalue()

            sample = {
                "__key__": subtile_name,
                f"rgb.{suffix}": im_byte_arr,
            }
            samples.append(sample)
    return samples


def split_inference_tiles(images, workers: int, **kwargs) -> None:
    """Split inference tiles into subtiles in parallel and save them to disk"""

    with wds.ShardWriter(
        str(kwargs["outdir"]) + "/inference/inference-%06d.tar", maxcount=512
    ) as sink:
        # process images in subsets
        for subset in [images[x : x + 100] for x in range(0, len(images), 100)]:
            data = process_map(
                partial(_split_inference_tile, **kwargs),
                subset,
                max_workers=workers,
                chunksize=1,
            )

            for sample in reduce(lambda z, y: z + y, data):
                if sample:
                    sink.write(sample)


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
        default=8192,
        help="size of input tiles [def: %(default)s]",
    )

    parser.add_argument(
        "--tile_size",
        dest="tile_size",
        type=int,
        default=512,
        help="size of final tiles that are then passed to the model [def: %(default)s]",
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
        "--all",
        dest="inference_tiles",
        default=False,
        action="store_true",
        help="also produce subtiles for inference-only tiles",
    )

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    Path(args.outdir / "train").mkdir(parents=True, exist_ok=True)

    if args.inference_tiles:
        Path(args.outdir / "inference").mkdir(parents=True, exist_ok=True)

    # subtile_stats = split_tiles(train_files)
    images = sorted(args.image_dir.glob("*.tif"))
    masks = sorted(args.mask_dir.glob("*.tif"))

    image_names = {i.name for i in images}
    mask_names = {i.name for i in masks}

    # limit set of images to images that have equivalent mask tiles
    train_images = [i for i in images if i.name in image_names.intersection(mask_names)]

    # images without masks (used for inference only)
    inference_images = [
        i for i in images if i.name in image_names.difference(mask_names)
    ]

    cfg = dict(
        source_dim=args.source_dim,
        tile_size=args.tile_size,
        outdir=args.outdir,
        format=args.format,
    )

    if args.inference_tiles:
        split_inference_tiles(inference_images, args.workers, **cfg)

    subtile_stats = split_tiles(train_images, masks, args.workers, **cfg)

    with open(args.outdir / "stats.csv", "w") as fout:
        fout.write("tile,deadtreepercent\n")
        for i, (fname, pct) in enumerate(subtile_stats):
            line = f"{fname},{pct}\n"
            fout.write(line)


if __name__ == "__main__":
    main()
