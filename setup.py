import sys

from deadtrees import logger
from deadtrees.constants import HOME_HTML, MODEL_CHECKPOINT_PATH, PACKAGE_DIR
from deadtrees.version import __version__
from setuptools import find_packages, setup

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
        "dvc[s3]",
        "python-dotenv",
        "fastapi",
        "hydra-core",
        "pydantic",
        "torch",
        "pytorch-lightning",
        "pytorch-lightning-bolts",
        "rioxarray",
        "tqdm",
        "webdataset",
        "xarray",
    ],
    extras_require={"train": ["wandb"], "preprocess": ["gdal", "pygeos"]},
    entry_points={
        "demo": [
            "deadtrees=deadtrees.__main__:main",
        ],
    },
    package_data={
        "deadtrees": [
            # str(MODEL_CHECKPOINT_PATH.relative_to(PACKAGE_DIR) / "*.torch"),
            # str(HOME_HTML.relative_to(PACKAGE_DIR)),
        ]
    },
)
