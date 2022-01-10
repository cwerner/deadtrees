import argparse
import math
from pathlib import Path

import numpy as np
import rioxarray
import torch
from deadtrees.data.deadtreedata import val_transform
from deadtrees.deployment.inference import PyTorchInference
from deadtrees.deployment.tiler import Tiler
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)

    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=Path,
        default=Path("checkpoints/bestmodel.ckpt"),
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
        "--all",
        action="store_true",
        dest="all",
        default=False,
        help="process complete directory",
    )

    parser.add_argument(
        "--nopreview",
        action="store_false",
        dest="preview",
        default=True,
        help="produce preview images",
    )

    args = parser.parse_args()

    bs = 64

    INFILE = args.infile

    def is_valid_tile(infile):
        with rioxarray.open_rasterio(infile).sel(band=1) as t:
            return False if np.isin(t, [0, 255]).all() else True

    # inference = ONNXInference("checkpoints/bestmodel.onnx")
    inference = PyTorchInference(args.model)

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

        batches = tiler.get_batches()
        batches = np.array_split(batches, math.ceil(len(batches) / bs), axis=0)

        out_batches = []

        for b, batch in enumerate(tqdm(batches, desc=INFILE.name)):
            batch_tensor = torch.stack(
                [val_transform(image=i.transpose(1, 2, 0))["image"] for i in batch]
            )

            # pytorch
            out_batch = (
                inference.run(batch_tensor.detach().to("cuda"), device="cuda")
                .cpu()
                .numpy()
            )

            out_batches.append(out_batch)

        OUTFILE = args.outpath / INFILE.name
        OUTFILE_PREVIEW = Path(str(args.outpath) + "_preview") / INFILE.name

        tiler.put_batches(np.concatenate(out_batches, axis=0))
        tiler.write_file(OUTFILE)

        if args.preview:
            image = Image.fromarray(np.uint8(tiler._target.values * 255), "L")
            image.save(OUTFILE_PREVIEW)


if __name__ == "__main__":
    main()
