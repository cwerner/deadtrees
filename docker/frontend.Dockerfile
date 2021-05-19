FROM python:3.8-slim

RUN mkdir /frontend

COPY docker/requirements-frontend.txt /frontend/requirements.txt

WORKDIR /frontend

RUN pip install -r requirements.txt

COPY . /frontend

EXPOSE 8501

CMD ["streamlit", "run", "deadtrees/deployment/ui.py"]
