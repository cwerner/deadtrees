import argparse
import io
from pathlib import Path

import webdataset as wds

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
    parser.add_argument("datapath", type=Path)
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    print(f"Scanning path: {Path(args.datapath)}")

    files = [str(x) for x in sorted(args.datapath.glob("*.tar"))]
    print("\nParsing files:")
    print(files)

    dataset = (
        wds.WebDataset(files)
        .map(sample_decoder)
        .rename(image="rgb.png", mask="msk.png", stats="txt")
        .map_dict(image=transform)
        .to_tuple("image")
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean, std = torch.zeros(3), torch.zeros(3)

    print("\nCalculating MEAN")
    for i, data in enumerate(dataloader):
        if i % 1000 == 0:
            print(i, end=" ", flush=True)
        data = data[0].squeeze(0)
        if i == 0:
            size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= i + 1

    mean_unsqueezed = mean.unsqueeze(1).unsqueeze(2)

    print("\nCalculating STD")
    for i, data in enumerate(dataloader):
        if i % 1000 == 0:
            print(i, end=" ", flush=True)
        data = data[0].squeeze(0)
        std += ((data - mean_unsqueezed) ** 2).sum((1, 2)) / size

    std /= i + 1
    std = std.sqrt()

    print(f"\nMean: {mean.tolist()}")
    print(f"STD: {std.tolist()}")


if __name__ == "__main__":
    main()
