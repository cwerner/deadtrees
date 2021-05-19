<div align="center">

# DeadTrees

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/static/v1?message=FastAPI&color=009688&logo=FastAPI&logoColor=FFFFFF&label="></a>
<a href="https://streamlit.io"><img alt="Streamlit" src="https://img.shields.io/static/v1?message=Streamlit&color=FF4B4B&logo=Streamlit&logoColor=FFFFFF&label="></a>

<br>

</div>

## Description
Map dead trees from ortho photos. A Unet (semantic segmentation model) is trained on a ortho photo collection of Luxembourg (year: 2019). This repository contains the preprocessing pipeline, training scripts, models, and a docker-based demo app (backend: FastAPI, frontend: Streamlit).

<p align="center">
   <img src="./assets/frontend.png" alt="Streamlit frontend"/>
   Fig 1: Streamlit UI for interactive prediction of dead trees in ortho photos.
</p>


## How to run

```yaml
# clone project
git clone https://github.com/cwerner/deadtrees
cd deadtrees

# [OPTIONAL] create virtual environment (using venve, pyenv, etc.) 
# and activate it

# install requirements (basic requirements):
pip install -e . 

# [OPTIONAL] install extra requirements for training:
pip install -e ".[train]"

# [OPTIONAL] install extra requirements to preprocess the raw data
# (instead of reading preprocessed data from S3):
pip install -e ".[preprocess]"

# [ALTERNATIVE] install all subpackages:
pip install -e ".[all]"
```

Train model with default configuration:
```yaml
cd scripts
python train.py
```

<br>
