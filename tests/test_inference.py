import io

import pytest
import webdataset as wds

import numpy as np
import torch
from deadtrees.deployment.inference import ONNXInference, PyTorchInference
from PIL import Image

# test_data = "tests/testdata/testdata.tar"  # 13 samples
test_data = "tests/testdata/train-balanced-short-000000.tar"  # 128 samples

bs = 4


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


@pytest.fixture
def sample_image():
    ds = (
        wds.Dataset(test_data)
        .map(semsegment_decoder)
        .rename(image="rgb.png", mask="msk.png")
        .to_tuple("image", "mask")
    )
    sample = next(iter(ds))
    return sample[0]


@pytest.fixture
def sample_batch():
    ds = (
        wds.Dataset(test_data)
        .map(semsegment_decoder)
        .rename(image="rgb.png", mask="msk.png")
        .to_tuple("image", "mask")
        .batched(bs)
    )
    sample = next(iter(ds))
    return sample[0]


@pytest.fixture
def pytorch_inference():
    return PyTorchInference("checkpoints/bestmodel.ckpt")


@pytest.fixture
def onnx_inference():
    return ONNXInference("checkpoints/bestmodel.onnx")


def test_inference_single_size(sample_image):
    assert sample_image.shape == (3, 512, 512)


def test_inference_batch_size(sample_batch):
    assert sample_batch.shape == (bs, 3, 512, 512)


# sizes: bs, y, x
def test_inference_pytorch_single_predict_size(pytorch_inference, sample_image):
    assert (pytorch_inference.run(sample_image)).shape == (512, 512)


def test_inference_pytorch_batch_predict_size(pytorch_inference, sample_batch):
    assert (pytorch_inference.run(sample_batch)).shape == (bs, 512, 512)


def test_inference_onnx_single_predict_size(onnx_inference, sample_image):
    sample_image_numpy = sample_image.detach().cpu().numpy()
    assert (onnx_inference.run(sample_image_numpy)).shape == (512, 512)


def test_inference_onnx_batch_predict_size(onnx_inference, sample_batch):
    sample_batch_numpy = sample_batch.detach().cpu().numpy()
    assert (onnx_inference.run(sample_batch_numpy)).shape == (bs, 512, 512)


# def test_inference_pytorch_batch():
#     pass


# def test_inference_onnx_single():
#     pass


# def test_inference_onnx_batch():
#     pass
