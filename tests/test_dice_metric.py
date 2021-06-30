import pytest
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

import numpy as np
import torch

n = 5  # w/h size
b = 1  # batch size
sample = torch.zeros((b, 2, n, n))  # BxCxHxW
sample[:, 0, :, :] = 1
sample[:, 0, 2:n, 2:n] = 0
sample[:, 1, 2:n, 2:n] = 1

increments = [(2, 1.0), (3, 0.7401), (4, 0.5)]
increments2 = [(2, 1.0), (3, 0.6154), (4, 0.2)]


@pytest.mark.parametrize("inc,res", increments)
def test_dicemetric_with_background(inc, res):

    fake_pred = torch.zeros((b, 2, n, n))  # BxCxHxW
    fake_pred[:, 0, :, :] = 1
    fake_pred[:, 0, inc:n, inc:n] = 0
    fake_pred[:, 1, inc:n, inc:n] = 1

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    score, _ = dice_metric(
        y_pred=fake_pred,
        y=sample,
    )

    result = torch.tensor([res])
    torch.testing.assert_allclose(score, result)


@pytest.mark.parametrize("inc,res", increments2)
def test_dicemetric_without_background(inc, res):

    fake_pred = torch.zeros((b, 2, n, n))  # BxCxHxW
    fake_pred[:, 0, :, :] = 1
    fake_pred[:, 0, inc:n, inc:n] = 0
    fake_pred[:, 1, inc:n, inc:n] = 1

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    score, _ = dice_metric(
        y_pred=fake_pred,
        y=sample,
    )

    result = torch.tensor([res])
    torch.testing.assert_allclose(score, result)


def test_dicemetric_all_zeros():

    sample = torch.zeros((b, 2, n, n))  # BxCxHxW
    sample[:, 0, :, :] = 1
    sample[:, 1, :, :] = 0

    fake_pred = torch.zeros((b, 2, n, n))  # BxCxHxW
    fake_pred[:, 0, :, :] = 1
    fake_pred[:, 0, 4:n, 4:n] = 0
    fake_pred[:, 1, 4:n, 4:n] = 1

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    score, _ = dice_metric(
        y_pred=fake_pred,
        y=sample,
    )

    torch.testing.assert_allclose(score, torch.tensor([0.9795918464660645]))


# def test_dicemetric_without_background(x):
#     assert x == 1
