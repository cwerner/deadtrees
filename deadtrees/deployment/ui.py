import io
import textwrap
from enum import Enum

import requests
import streamlit as st
from models import PredictionStats
from requests_toolbelt.multipart.encoder import MultipartEncoder

from PIL import Image

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

itype = st.selectbox(
    "Inference type", list(inf_types.keys()), format_func=inf_types.get
)


input_image = st.file_uploader("Insert Image")  # image upload widget

if st.button("Get Segmentation Map"):

    col1, col2 = st.beta_columns(2)

    if input_image:
        result = process(input_image, f"{backend}?model_type={itype.value}")

        rgb_image = Image.open(input_image).convert("RGB")
        mask_image = Image.open(io.BytesIO(result["mask"])).convert("RGB")

        col1.header("Source")
        col1.image(rgb_image, use_column_width=True)
        col2.header("Prediction")
        col2.image(mask_image, use_column_width=True)

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
