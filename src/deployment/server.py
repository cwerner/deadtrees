import io
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, File
from pydantic import BaseModel
from starlette.responses import HTMLResponse, Response

import numpy as np
import torch
from numpy.lib.arraysetops import isin
from PIL import Image
from src.data.deadtreedata import val_transform
from src.deployment.inference import ONNXInference, PyTorchInference
from src.deployment.models import PredictionStats, predictionstats_to_str
from src.utils.timer import record_execution_time

MODEL = "bestmodel"

# TODO: make this an endpoint
pytorch_model = PyTorchInference(f"checkpoints/{MODEL}.ckpt")
onnx_model = ONNXInference(f"checkpoints/{MODEL}.onnx")

app = FastAPI(
    title="DeadTrees image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via our UNet implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """\
    <!doctype html>
    <html lang="en">
        <head>
            <!-- Required meta tags -->
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

            <!-- Bootstrap CSS -->
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

            <title>DeadTrees Inference API</title>
            <meta http-equiv="refresh" content="7; URL=./docs" />
        </head>
        <body>

        <div class="d-flex vh-100">
            <div class="d-flex w-100 justify-content-center align-self-center">
                <div class="jumbotron">
                    <h1 class="display-4">🌲☠️🌲🌲🌲 DeadTrees Inference API 🌲🌲☠️☠️🌲</h1>
                    <p class="lead">REST API for semantic segmentation of dead trees from ortho photos</p>
                    <hr class="my-4">
                    <p>
                    There also is an <a href="./" onmouseover="javascript:event.target.port=8502">interactive streamlit frontend</a>. You will be redirected to the <a href="./docs"><b>OpenAPI documentation page</b></a> in 10 seconds.
                    </p>
                </div>
            </div>
        </div>

        <!-- Optional JavaScript -->
        <!-- jQuery first, then Popper.js, then Bootstrap JS -->
        <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>

        </body>
    </html>
    """


class ModelTypes(Enum):
    """allowed model types"""

    PYTORCH = "pytorch"
    ONNX = "onnx"


def split_image_into_tiles(image: Image):
    # complete this: what about batches?
    batch = val_transform(image=image)["image"]
    return batch


@app.post("/segmentation")
def get_segmentation_map(
    file: bytes = File(...), model_type: Optional[ModelTypes] = None
):
    """Get segmentation maps from image file"""

    model_type = model_type or ModelTypes.PYTORCH

    image = Image.open(io.BytesIO(file)).convert("RGB")
    input_tensor = val_transform(image=np.array(image))["image"]

    # call prediction and measure execution time
    with record_execution_time() as elapsed:
        if model_type == ModelTypes.PYTORCH:
            out = pytorch_model.run(input_tensor)
        elif model_type == ModelTypes.ONNX:
            out = onnx_model.run(input_tensor.detach().cpu().numpy())
        else:
            raise ValueError("only pytorch and onnx models allowed")

    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()

    # TODO: compose batch if required
    image = Image.fromarray(np.uint8(out * 255), "L")
    dead_tree_fraction = float(out.sum() / out.size)

    stats = PredictionStats(
        fraction=dead_tree_fraction,
        model_name=MODEL,
        model_type=model_type.value,
        elapsed=elapsed(),
    )

    bytes_io = io.BytesIO()
    image.save(bytes_io, format="PNG")

    return Response(
        bytes_io.getvalue(),
        headers=predictionstats_to_str(stats),
        media_type="image/png",
    )
