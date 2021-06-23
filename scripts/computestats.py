import argparse
import io
import itertools
from pathlib import Path

import webdataset as wds

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


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
        img = img.convert("RGB")
    return np.array(img)


def sample_decoder(sample, img_suffix="rgb.png"):
    """Decode data triplet (image, mask stats) from sharded datastore"""

    assert img_suffix in sample, "Wrong image suffix provided"
    sample[img_suffix] = image_decoder(sample[img_suffix])
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=Path, nargs="+")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    print(f"Scanning path(s): {args.datapath}")

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

    if len(tar_files) > len(tif_files):
        # webdataset
        dataset = (
            wds.WebDataset([str(x) for x in tar_files])
            .map(sample_decoder)
            .rename(image="rgb.png", mask="msk.png", stats="txt")
            .map_dict(image=transform)
            .to_tuple("image")
        )
    else:
        # plain source tif dataset
        dataset = TifDataset(tif_files, transform=transform)
        # ,
        #     is_valid_file=lambda x: Path(x).suffix == ".tif"
        #     )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean, std = torch.zeros(3), torch.zeros(3)

    print("\nCalculating MEAN")
    for i, data in enumerate(tqdm(dataloader)):
        data = data[0].squeeze(0)
        if i == 0:
            size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= i + 1

    mean_unsqueezed = mean.unsqueeze(1).unsqueeze(2)

    print("\nCalculating STD")
    for i, data in enumerate(tqdm(dataloader)):
        data = data[0].squeeze(0)
        std += ((data - mean_unsqueezed) ** 2).sum((1, 2)) / size

    std /= i + 1
    std = std.sqrt()

    print(f"\nMean: {mean.tolist()}")
    print(f"STD: {std.tolist()}")


if __name__ == "__main__":
    main()
