import io
import textwrap

import requests
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder

from PIL import Image

# interact with FastAPI endpoint
backend = "http://backend:8000/segmentation"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("DeadTree image segmentation")

st.write(
    """Obtain semantic segmentation maps of the image in input via our UNet implemented in PyTorch.
         Visit this URL at port 8501 for the streamlit interface."""
)  # description and instructions


input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Get segmentation map"):

    col1, col2 = st.beta_columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Segmented")
        col2.image(segmented_image, use_column_width=True)

        d = segments.headers
        st.markdown(
            textwrap.dedent(
                f"""\
                #### Prediction stats  
                Model used: **{d['model_name']}**  
                Percentage of dead trees detected: **{(float(d['fraction'])*100):.2f}%** 
                Inference duration: **{float(d['elapsed']):.1f}sec**  
                """  # noqa
            )
        )
    else:
        # handle case with no image
        st.write("Insert an image!")
