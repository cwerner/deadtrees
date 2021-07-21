import logging
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from deadtrees.data.deadtreedata import DeadtreeDatasetConfig
from matplotlib.offsetbox import AnchoredText
from skimage.io import imread

plt.style.use("ggplot")


logger = logging.getLogger(__name__)


def is_running_from_ipython() -> bool:
    """Check if code is executed in ipython (jupyter?) environment"""
    from IPython import get_ipython

    return get_ipython() is not None


def render_image(a: np.ndarray, width=600) -> None:
    """Display ndarray in rgb image format in Jupyter"""

    # TODO: simplify?
    from io import BytesIO

    import IPython

    from PIL import Image

    img_crop_pil = Image.fromarray(a)

    with BytesIO() as byte_io:
        img_crop_pil.save(byte_io, format="png")
        png_buffer = byte_io.getvalue()

    i = IPython.display.Image(data=png_buffer, width=width)
    IPython.display.display(i)


def fig2img(fig: plt.Figure, dpi: int = 72) -> np.ndarray:
    """Convert a Matplotlib figure to a PIL Image and return it"""

    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return imread(buf)


def rgbtensor_to_rgb(x: torch.Tensor) -> List[np.array]:
    """Inverse Normalize a rgb tensor and return list of rgb samples"""
    MEAN, STD = DeadtreeDatasetConfig.mean, DeadtreeDatasetConfig.std

    x_norm = []
    for i in range(len(x)):
        x_norm.append(
            np.array((x[i].permute(1, 2, 0) * STD + MEAN) * 255, dtype="uint8")
        )
    return x_norm


def masktensor_to_rgb(
    x: torch.Tensor, base_rgb: Optional[torch.Tensor] = None
) -> List[np.array]:
    """Scale mask tensor to rgb sample"""

    rgbs = rgbtensor_to_rgb(base_rgb) if base_rgb is not None else None

    x_norm = []
    for i in range(len(x)):
        mask = np.array(x[i].unsqueeze(dim=0).permute(1, 2, 0))
        mask2 = np.array(np.where(mask == 1, 1, 0.66) * 255, dtype="uint8")
        x_norm.append(np.dstack([rgbs[i], mask2]) if base_rgb is not None else mask)
    return x_norm


def show(
    x: Union[torch.Tensor, np.ndarray],
    *,
    y: Optional[Union[torch.Tensor, np.ndarray]] = None,
    y_hat: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_samples: Optional[int] = 1,
    stats: Optional[Dict] = None,
    dpi: Optional[int] = 100,
    display: Optional[bool] = False,
) -> np.ndarray:

    items = {k: v for k, v in zip(["x", "y", "y_hat"], [x, y, y_hat]) if v is not None}
    items_orig = items.copy()

    # check for identical types
    if not all(isinstance(i, type(x)) for i in items.values()):
        raise ValueError("types of x, y, (y_hat) have to be identical")

    # move to cpu if necessary
    if isinstance(items["x"], torch.Tensor):
        items = {k: v.cpu() for k, v in items.items()}

        if y_hat is not None:
            items["y_hat"] = items["y_hat"].argmax(dim=1)

    items["x"] = rgbtensor_to_rgb(items["x"])

    if "y" in items:
        # fancy_x_rgb, y_rgb = masktensor_to_rgb(items['y'], base_rgb=items_orig['x'])
        # items['x'] = fancy_x_rgb
        # items['y'] = y_rgb
        items["x_masked"] = masktensor_to_rgb(items["y"], base_rgb=items_orig["x"])

    # plotting
    subplot_size = 4  # plt in inches

    n_samples = len(x) if not n_samples else n_samples

    fig_size_x = subplot_size * len(items)
    fig_size_y = subplot_size * n_samples

    fig, ax = plt.subplots(
        n_samples,
        len(items),
        figsize=(fig_size_x, fig_size_y),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    if isinstance(ax, matplotlib.axes.Axes):
        ax = np.array([ax])

    if len(ax.shape) == 1:
        ax = ax[np.newaxis, :]

    for i in range(n_samples):
        for j, k in enumerate(sorted(items.keys())):
            im = items[k]
            ax[i, j].imshow(im[i])

            if k == "x" and stats:
                txt = f"DTF: {stats[i]['frac']:.2f} [%]"
                anchored_text = AnchoredText(
                    txt, loc=2, prop=dict(fontsize=8, color="purple")
                )
                ax[i, j].add_artist(anchored_text)

            if k == "x":
                ax[0, j].set_title(r"$X$")
            if k == "x_masked":
                ax[0, j].set_title(r"$X_{mask}$")
            if k == "y":
                ax[0, j].set_title(r"$y$")
            if k == "y_hat":
                ax[0, j].set_title(r"$\hat{y}$")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    for i in range(n_samples):
        if stats:
            ax[i, 0].set_ylabel(stats[i]["file"], fontsize=8)
        else:
            ax[i, 0].set_ylabel(f"Sample {i}")

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    out = fig2img(fig, dpi=dpi)
    plt.close(fig)

    if is_running_from_ipython():
        if display:
            render_image(out)
            return None

    return out
