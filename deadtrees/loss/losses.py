# source: https://github.com/LIVIAETS/boundary-loss
# paper: https://doi.org/10.1016/j.media.2020.101851
# license: unspecified as of 2021-12-06

import argparse
from functools import partial
from multiprocessing.pool import Pool
from operator import add
from pathlib import Path
from random import randint, random, uniform
from typing import Any, Callable, cast, Iterable, List, Set, Tuple, TypeVar, Union

from scipy.ndimage import distance_transform_edt as eucl_distance

import numpy as np
import torch
import torch.sparse
from PIL import Image, ImageOps
from skimage.io import imsave
from torch import einsum, Tensor
from tqdm import tqdm

EPS = 1e-10


A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for li in to_flat for e in li]


def flatten__(to_flat):
    if type(to_flat) != list:
        return [to_flat]

    return [e for li in to_flat for e in flatten__(li)]


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0


# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->bk", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bk...->k", a)[..., None]


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    return True
    # _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    # _ones = torch.ones_like(_sum, dtype=torch.float32)
    # return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(
    sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8
) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(
        torch.float32
    )
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(
        torch.float32
    )

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bk...->bk", union(a, b).type(torch.float32))


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    return class2one_hot(torch.from_numpy(seg.copy()).type(torch.int64), K).numpy()


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(
    seg: np.ndarray, resolution: Tuple[float, float, float] = None, dtype=None
) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = (
                eucl_distance(negmask, sampling=resolution) * negmask
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
            )
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def one_hot2hd_dist(
    seg: np.ndarray, resolution: Tuple[float, float, float] = None, dtype=None
) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            res[k] = eucl_distance(posmask, sampling=resolution)

    return res


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    """ center cropping """

    def g_center(arr):
        if arr.shape == shape:
            return arr

        offsets: List[int] = [(arrs - s) // 2 for (arrs, s) in zip(arr.shape, shape)]

        if 0 in offsets:
            return arr[[slice(0, s) for s in shape]]

        res = arr[[slice(d, -d) for d in offsets]][
            [slice(0, s) for s in shape]
        ]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, offsets)

        return res

    return [g_center(arr) for arr in arrs]


def center_pad(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    assert len(arr.shape) == len(target_shape)

    diff: List[int] = [(nx - x) for (x, nx) in zip(arr.shape, target_shape)]
    pad_width: List[Tuple[int, int]] = [(w // 2, w - (w // 2)) for w in diff]

    res = np.pad(arr, pad_width)
    assert res.shape == target_shape, (res.shape, target_shape)

    return res


# ----------------------


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        # modification: move EPS outside to reduce risk of zero-division
        # orig: w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + EPS) ** 2)
        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) ** 2) + EPS)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + EPS) / (
            einsum("bk->b", union) + EPS
        )

        loss = divided.mean()

        return loss


class DiceLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc)

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + EPS) / (
            union + EPS
        )

        loss = divided.mean()

        return loss


class SurfaceLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


BoundaryLoss = SurfaceLoss


class FocalLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + EPS).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs) ** self.gamma
        loss = -einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
        loss /= mask.sum() + EPS

        return loss
