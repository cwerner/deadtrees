import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnchoredText
from skimage.io import concatenate_images, imread, imshow
from skimage.morphology import label
from skimage.transform import resize

plt.style.use("ggplot")


logger = logging.getLogger(__name__)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=72)
    buf.seek(0)
    return imread(buf)


def show(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    y_hat: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_samples: Optional[int] = 4,
    threshold: Optional[float] = 0.95,
    stats: Optional[Dict] = None,
):
    """
    show: helper function to display rgb sample (X), target mask (y) and predicted mask (yhat)
    """

    # check for identical types
    items = [x, y, y_hat] if y_hat is not None else [x, y]
    if not all(isinstance(i, type(x)) for i in items):
        raise ValueError("types of x, y, (y_hat) have to be identical")

    # check for identical batch sizes
    if not all(len(i) == len(items[0]) for i in items):
        raise ValueError("sizes of x, y, (y_hat) have to be identical")

    if isinstance(x, torch.Tensor):
        x = x.cpu()
        y = y.cpu()
        if y_hat is not None:
            y_hat = y_hat.cpu().argmax(dim=1)

    subplot_size = 4  # plt in inches

    # n_samples = n_samples if len(x) > n_samples else len(x)
    n_samples = len(x) if not n_samples or len(x) > n_samples else n_samples

    fig_size_x = subplot_size * (len(items) + 1)
    fig_size_y = subplot_size * n_samples

    fig, ax = plt.subplots(
        n_samples,
        len(items) + 1,
        figsize=(fig_size_y, fig_size_x),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]

    for i in range(n_samples):
        rgb = np.array(x[i].permute(1, 2, 0) * 255, dtype="uint8")

        mask = np.array(y[i].unsqueeze(dim=0).permute(1, 2, 0))
        mask2 = np.array(np.where(mask == 1, 1, 0.66) * 255, dtype="uint8")
        combo = np.dstack([rgb, mask2])

        ax[i, 0].imshow(rgb)

        if stats:
            txt = f"DTF: {stats['frac'][i]:.2f} [%]"
            anchored_text = AnchoredText(
                txt, loc=2, prop=dict(fontsize=10, color="purple")
            )
            ax[i, 0].add_artist(anchored_text)

        ax[i, 1].imshow(combo)
        ax[i, 2].imshow(mask)

        if y_hat is not None:
            # prediction = np.where(y_hat[i] >= threshold, 1, 0)
            # prediction = np.array(y_hat[i] * 255, dtype='uint8')
            ax[i, 3].imshow(y_hat[i])

    ax[0, 0].set_title(r"$X$")
    ax[0, 1].set_title(r"$X_{masked}$")
    ax[0, 2].set_title(r"$y$")
    if y_hat is not None:
        ax[0, 3].set_title(r"$\hat{y}$")

    plt.setp(ax, xticks=[], yticks=[])

    for i in range(n_samples):
        if stats:
            ax[i, 0].set_ylabel(stats["filename"][i], fontsize=10)
        else:
            ax[i, 0].set_ylabel(f"Sample {i}")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    out = fig2img(fig)
    plt.close(fig)
    return out
