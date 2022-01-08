from pathlib import Path

from src.checkpoint import CHECKPOINT

# local directories
PACKAGE_DIR = Path(__file__).parent.absolute()
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"

HOME_HTML = PACKAGE_DIR / "api" / "index.html"

MODEL_CHECKPOINT_PATH = MODELS_DIR / CHECKPOINT
