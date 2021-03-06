import sys

from setuptools import find_packages, setup

from src import logger
from src.constants import HOME_HTML, MODEL_CHECKPOINT_PATH, PACKAGE_DIR
from src.version import __version__

if not MODEL_CHECKPOINT_PATH.exists():
    # develop will be in argv if we do e.g. `pip install -e .`
    if "develop" not in sys.argv:
        # logger.error("can't build a non-development package with no model")
        # raise FileNotFoundError(MODEL_CHECKPOINT_PATH)
        pass
setup(
    name="deadtrees",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "bottleneck",
        "dask",
        "fastapi",
        "hydra-core",
        "pydantic",
        "torch",
        "pytorch-lightning",
        "pytorch-lightning-bolts",
        "rioxarray",
        "tqdm",
        "xarray",
    ],
    extras_require={"train": ["wandb"]},
    entry_points={
        "demo": [
            "deadtrees=src.__main__:main",
        ],
    },
    package_data={
        "deadtrees": [
            # str(MODEL_CHECKPOINT_PATH.relative_to(PACKAGE_DIR) / "*.torch"),
            # str(HOME_HTML.relative_to(PACKAGE_DIR)),
        ]
    },
)
