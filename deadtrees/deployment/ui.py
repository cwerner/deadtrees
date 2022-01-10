import io
import pathlib
import textwrap
from enum import Enum

import requests
import streamlit as st
import streamlit.components.v1 as components
from models import PredictionStats
from requests_toolbelt.multipart.encoder import MultipartEncoder

from PIL import Image


# Source: https://github.com/robmarkcole/streamlit-image-juxtapose.git
def juxtapose(img1: str, img2: str, height: int = 1000):  # data

    """Create a new timeline component.
    Parameters
    ----------
    height: int or None
        Height of the timeline in px
    Returns
    -------
    static_component: Boolean
        Returns a static component with a timeline
    """

    # load css + js
    cdn_path = "https://cdn.knightlab.com/libs/juxtapose/latest"
    css_block = f'<link rel="stylesheet" href="{cdn_path}/css/juxtapose.css">'
    js_block = f'<script src="{cdn_path}/js/juxtapose.min.js"></script>'

    # write html block
    htmlcode = (
        css_block
        + """
    """
        + js_block
        + """
        <div id="foo" style="width: 95%; height: """
        + str(height)
        + '''px; margin: 1px;"></div>
        <script>
        slider = new juxtapose.JXSlider('#foo',
            [
                {
                    src: "'''
        + img1
        + '''",
                    label: 'source',
                },
                {
                    src: "'''
        + img2
        + """",
                    label: 'prediction',
                }
            ],
            {
                animate: true,
                showLabels: true,
                showCredits: true,
                startingPosition: "50%",
                makeResponsive: true
            });
        </script>
    """
    )
    static_component = components.html(
        htmlcode,
        height=height,
    )
    return static_component


STREAMLIT_STATIC_PATH = (
    pathlib.Path(st.__path__[0]) / "static"
)  # at venv/lib/python3.9/site-packages/streamlit/static

# interact with FastAPI endpoint
backend = "http://backend:8000/segmentation"


# TDOD: refactor to central localtion
class ModelTypes(Enum):
    """allowed model types"""

    PYTORCH = "pytorch"
    ONNX = "onnx"


def process(image: bytes, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return {
        "mask": r.content,
        "stats": PredictionStats.parse_obj(r.headers),
    }


# construct UI layout
st.title("DeadTree image segmentation")

st.write(
    """Obtain semantic segmentation maps of the image in input via our UNet implemented in PyTorch.
         Visit this URL at port 8000 for REST API."""
)  # description and instructions


inf_types = {
    ModelTypes.PYTORCH: "PyTorch (native)",
    ModelTypes.ONNX: "ONNX",
}

col1, col2 = st.beta_columns(2)

itype = col1.selectbox(
    "Inference type", list(inf_types.keys()), format_func=inf_types.get
)

vtype = col2.radio("Display", ("Side-by-side", "Slider"), index=1)


input_image = st.file_uploader("Insert Image")  # image upload widget

if st.button("Get Segmentation Map"):

    if input_image:

        result = process(input_image, f"{backend}?model_type={itype.value}")

        rgb_image = Image.open(input_image).convert("RGB")
        mask_image = Image.open(io.BytesIO(result["mask"])).convert("RGB")

        if vtype == "Side-by-side":
            col1, col2 = st.beta_columns(2)
            col1.header("Source")
            col1.image(rgb_image, use_column_width=True)
            col2.header("Prediction")
            col2.image(mask_image, use_column_width=True)

        else:
            IMG1 = "source.png"
            IMG2 = "prediction.png"
            rgb_image.save(STREAMLIT_STATIC_PATH / IMG1)
            mask_image.save(STREAMLIT_STATIC_PATH / IMG2)
            juxtapose(IMG1, IMG2, height=600)

        stats = result["stats"]
        st.markdown(
            textwrap.dedent(
                f"""\
                ### Stats 📊   
                Model: **{stats.model_name}**  
                Format: **{stats.model_type}**  
                Percentage of dead trees detected: **{stats.fraction*100:.2f}%**  
                Inference duration: **{stats.elapsed:.1f}sec**  
                """  # noqa
            )
        )
    else:
        # handle case with no image
        st.write("Insert an image!")
