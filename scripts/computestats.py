import argparse
import datetime
import io
import itertools
import json
from pathlib import Path

import webdataset as wds

import numpy as np
import pandas as pd
import PIL
import torchvision.transforms as transforms
from deadtrees.utils.data_handling import make_blocks_vectorized
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


class TifDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)


def image_decoder(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("RGBA")
    return np.array(img)


def sample_decoder(sample, img_suffix="rgbn.tif"):
    """Decode data triplet (image, mask stats) from sharded datastore"""

    assert img_suffix in sample, "Wrong image suffix provided"
    sample[img_suffix] = image_decoder(sample[img_suffix])
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=Path, nargs="+")

    parser.add_argument(
        "--frac",
        dest="frac",
        type=float,
        default=1.0,
        help="fraction of tiles to consider [range: 0-1, def: %(default)s]",
    )

    args = parser.parse_args()

    np.random.seed(42)
    print("Using fixed random seed!")

    # constants
    tile_size = 256
    size = tile_size ** 2

    if isinstance(args.datapath, list):
        tar_files = sorted(
            list(itertools.chain(*[x.glob("*.tar") for x in args.datapath]))
        )
        tif_files = sorted(
            list(itertools.chain(*[x.glob("*.tif") for x in args.datapath]))
        )
    else:
        tar_files = sorted(args.datapath.glob("*.tar"))
        tif_files = sorted(args.datapath.glob("*.tif"))

    n_files = len(tif_files)

    SUBSET = int(round(args.frac * n_files, 0))
    selection = np.random.choice(range(n_files), size=SUBSET, replace=False)

    if len(tar_files) > len(tif_files):
        # webdataset
        dataset = (
            wds.WebDataset([str(x) for x in tar_files])
            .map(sample_decoder)
            .rename(image="rgbn.tif", mask="mask.tif", stats="txt")
            .map_dict(image=transform)
            .to_tuple("image")
        )
    else:
        # plain source tif dataset
        dataset = TifDataset(tif_files, transform=transform)

    dataset = Subset(dataset, selection)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean, std = np.zeros(4), np.zeros(4)

    print("\nCalculating STATS")

    print("\nCalculating MEAN")
    cnt = 0

    for i, data in enumerate(tqdm(dataloader)):
        data = data.squeeze(0).numpy()

        # ignore incomplete tiles for stats
        if data.shape[-2] != data.shape[-1]:
            continue

        # check for empty tile and skip (all values are either 0 or 1 in the first band):
        if np.isin(data, [0, 1]).all():
            continue

        subtiles_rgbn = make_blocks_vectorized(data, tile_size)
        for subtile_rgbn in subtiles_rgbn:
            if subtile_rgbn[0].min() != subtile_rgbn[0].max():
                mean += subtile_rgbn.sum((1, 2)) / size
                cnt += 1

    mean /= cnt + 1  # i + 1

    mean_unsqueezed = np.expand_dims(
        np.expand_dims(mean, 1), 2
    )  # mean.unsqueeze(1).unsqueeze(2)

    print("\nCalculating STD")
    cnt = 0
    for i, data in enumerate(tqdm(dataloader)):
        data = data.squeeze(0).numpy()

        # ignore incomplete tiles for stats
        if data.shape[-2] != data.shape[-1]:
            continue

        # check for empty tile and skip (all values are either 0 or 1 in the first band):
        if np.isin(data, [0, 1]).all():
            continue

        subtiles_rgbn = make_blocks_vectorized(data, tile_size)
        for subtile_rgbn in subtiles_rgbn:
            if subtile_rgbn[0].min() != subtile_rgbn[0].max():
                std += ((subtile_rgbn - mean_unsqueezed) ** 2).sum((1, 2)) / size
                cnt += 1

    std /= cnt + 1
    std = np.sqrt(std)  # std.sqrt()

    df = pd.DataFrame(
        {
            "band": ["red", "green", "blue", "nir"],
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    )
    df = df.set_index("band")

    # report
    info = {
        "sources": [str(x) for x in args.datapath],
        "date": str(datetime.datetime.now()),
        "frac": args.frac,
        "subtiles": cnt,
        "results": json.loads(df.to_json(orient="index")),
    }

    # Serializing json
    with open(args.datapath[0].parent / "processed.images.stats.json", "w") as fout:
        fout.write(json.dumps(info, indent=4))


if __name__ == "__main__":
    main()
