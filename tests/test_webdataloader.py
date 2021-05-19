# test the basic data loading stuff
#

import io

import webdataset as wds

import numpy as np
import torch
from PIL import Image

# test_data = "tests/testdata/testdata.tar"  # 13 samples
test_data = "tests/testdata/train-balanced-short-000000.tar"  # 128 samples


def count_samples_tuple(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


def test_dataset():
    ds = wds.Dataset(test_data)
    assert count_samples_tuple(ds) == 64


def test_dataset_shuffle_extract():
    ds = wds.Dataset(test_data).shuffle(5).to_tuple("msk.png rgb.png")
    assert count_samples_tuple(ds) == 64


def test_dataset_pipe_cat():
    ds = wds.Dataset(f"pipe:cat {test_data}").shuffle(5).to_tuple("msk.png rgb.png")
    assert count_samples_tuple(ds) == 64


def test_slice():
    ds = wds.Dataset(test_data).slice(10)
    assert count_samples_tuple(ds) == 10


def test_rename():
    ds = wds.Dataset(test_data).rename(image="rgb.png", mask="msk.png")
    sample = next(iter(ds))
    assert set(sample.keys()) == {"image", "mask"}


def test_torch_sample_decoder():
    def image_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("RGB")
        result = np.asarray(img)
        result = np.array(result.transpose(2, 0, 1))
        return torch.tensor(result) / 255.0

    def mask_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("L")
        result = np.asarray(img)
        return torch.tensor(result)

    def semsegment_decoder(sample):
        sample = dict(sample)
        sample["rgb.png"] = image_decoder(sample["rgb.png"])
        sample["msk.png"] = mask_decoder(sample["msk.png"])
        return sample

    ds = (
        wds.Dataset(test_data)
        .map(semsegment_decoder)
        .rename(image="rgb.png", mask="msk.png")
        .to_tuple("image", "mask")
    )

    image, mask = next(iter(ds))
    assert (image.shape, mask.shape) == ((3, 512, 512), (512, 512))


def test_torch_map_dict_decoder():
    def image_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("RGB")
        result = np.asarray(img)
        result = np.array(result.transpose(2, 0, 1))
        return torch.tensor(result) / 255.0

    def mask_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("L")
        result = np.asarray(img)
        return torch.tensor(result)

    ds = (
        wds.Dataset(test_data)
        .rename(image="rgb.png", mask="msk.png")
        .map_dict(image=image_decoder, mask=mask_decoder)
        .to_tuple("image", "mask")
    )

    image, mask = next(iter(ds))
    assert (image.shape, mask.shape) == ((3, 512, 512), (512, 512))


def test_torch_map_dict_batched_decoder():

    bs = 8

    def image_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("RGB")
        result = np.asarray(img)
        result = np.array(result.transpose(2, 0, 1))
        return torch.tensor(result) / 255.0

    def mask_decoder(data):
        with io.BytesIO(data) as stream:
            img = Image.open(stream)
            img.load()
            img = img.convert("L")
        result = np.asarray(img)
        return torch.tensor(result)

    ds = (
        wds.Dataset(test_data)
        .rename(image="rgb.png", mask="msk.png")
        .map_dict(image=image_decoder, mask=mask_decoder)
        .to_tuple("image", "mask")
        .batched(bs, partial=False)
    )

    image, mask = next(iter(ds))
    assert (image.shape, mask.shape) == ((bs, 3, 512, 512), (bs, 512, 512))
