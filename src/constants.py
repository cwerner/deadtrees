from pathlib import Path
from src.checkpoint import CHECKPOINT

# local directories
PACKAGE_DIR = Path(__file__).parent.absolute()

HOME_HTML = PACKAGE_DIR / "api" / "index.html"

MODEL_CHECKPOINT_PATH = MODELS_DIR / CHECKPOINT

