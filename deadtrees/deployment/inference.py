import io

import numpy as np
import torch
from deadtrees.data.deadtreedata import val_transform
from deadtrees.network.segmodel import SemSegment
from matplotlib import cm
from PIL import Image


def get_model(model_path: str = "bestmodel.ckpt"):
    model = SemSegment.load_from_checkpoint(model_path)
    model.eval()
    return model


def split_image_into_tiles(image: Image):
    # complete this: what about batches?
    batch = val_transform(image=image)["image"]
    return batch.unsqueeze(0)


def get_segmentation(
    model: SemSegment, binary_image: bytes, model_name: str = "unknown"
):
    image = Image.open(io.BytesIO(binary_image)).convert("RGB")

    batch = split_image_into_tiles(np.array(image))

    import time

    start = time.process_time()

    with torch.no_grad():
        output = model(batch)

    elapsed = time.process_time() - start

    output_predictions = output.argmax(1)

    image = Image.fromarray(np.uint8(output_predictions.squeeze() * 255), "L")
    dead_tree_fraction = (
        torch.count_nonzero(output_predictions) / torch.numel(output_predictions)
    ).item()

    return {
        "image": image,
        "stats": {
            "fraction": str(dead_tree_fraction),
            "model_name": model_name,
            "elapsed": str(elapsed),
        },
    }
