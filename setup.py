import sys
from itertools import chain

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

# extra package dependencies
EXTRAS = {
    "train": ["wandb"],
    "preprocess": [
        "gdal",
        "pygeos",
        "bottleneck",
        "dask",
        "rioxarray>=0.4",
        "xarray",
    ],
}
EXTRAS["all"] = [i for i in chain.from_iterable(EXTRAS.values())]


setup(
    name="deadtrees",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "albumentations",
        "dvc[s3]",
        "python-dotenv",
        "hydra-core>=1.1.0",
        "hydra-colorlog>=1.1.0",
        "pydantic",
        "torch>=1.8.1",
        "torchvision>=0.9.1",
        "pytorch-lightning>=1.3.7",
        "rich",
        "tqdm",
        "webdataset==0.1.62",
        "segmentation_models_pytorch>=0.2.0",
    ],
    # install in editable mode: pip install -e ".[train,preprocess]" or
    #                           pip install -e ".[all]"
    extras_require=EXTRAS,
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
