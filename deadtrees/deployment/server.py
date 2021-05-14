import io
from typing import Any, Dict

from fastapi import FastAPI, File
from pydantic import BaseModel
from starlette.responses import HTMLResponse, Response

from deadtrees.deployment.inference import get_model, get_segmentation

MODEL = "bestmodel.ckpt"

model = get_model(MODEL)

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
                    <p>You will be redirected to the <a href="./docs"><b>OpenAPI documentation page</b></a> in 7 seconds...</p>
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


@app.post("/segmentation")  # , response_model=Prediction)
def get_segmentation_map(file: bytes = File(...)):
    """Get segmentation maps from image file"""

    # return data dict, convert to pydantic model (?)
    data = get_segmentation(model, file, model_name=MODEL)

    bytes_io = io.BytesIO()
    data["image"].save(bytes_io, format="PNG")

    stats = {
        "fraction": data["stats"]["fraction"],
        "model_name": data["stats"]["model_name"],
        "elapsed": data["stats"]["elapsed"],
    }
    return Response(bytes_io.getvalue(), headers=stats, media_type="image/png")
