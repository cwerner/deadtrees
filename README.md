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

# [OPTIONAL] create virtual environment (using venve, pyenv, etc.) and activate it. An easy way to get a base system configured is to use micromamba (a faster alternative to anaconda) and the fastchan channel to install the notoriously finicky pytorch base dependencies and cuda setup

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# init shell
./bin/micromamba shell init -s bash -p ~/micromamba
source ~/.bashrc

micromamba create -p deadtrees python=3.9 -c conda-forge
micromamba activate deadtrees
micromamba install pytorch torchvision albumentations -c fastchan -c conda-forge

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

Download the dataset from S3 (output of the createdataset dvc stage)
```yaml
dvc pull createdataset
```

Specify the location of the training dataset on your system by creating the file  `.env` with the following syntax:
```
export TRAIN_DATASET_PATH="/path_to_my_repos/deadtrees/data/dataset/train"
```

Train model with default configuration (you can adjust the training config on the commandline or by editing the hydra yaml files in `conf`): 
```yaml
python scripts/train.py
```

<br>
