FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

# only for albumentations, remove when they cleanup their dependencies  
# https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6

RUN mkdir /backend

COPY docker/requirements-backend.txt /backend/requirements.txt

WORKDIR /backend

COPY . /backend

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html


EXPOSE 8000

CMD ["uvicorn", "deadtrees.deployment.server:app", "--host", "0.0.0.0", "--port", "8000"]
