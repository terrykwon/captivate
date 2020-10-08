# FROM python:3.8.5-buster
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# make sure this matches the bind mount volume in docker-compose!!!
WORKDIR /workspace/modelserver

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    gcc musl-dev ffmpeg \
    g++ default-jdk


COPY ./requirements.txt .
RUN pip install -r requirements.txt

# install local packages
COPY ./models ./models
# RUN pip install ./models/retinaface
RUN pip install ./models/visual-attention
# RUN pip install -e ./models/arcface

# install detectron2
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

# RUN python ./models/yolact/external/DCNv2/setup.py build develop

# still necessary to do this besides bind mount so that files are persisted
# this should be the last layer so that deeper layers aren't constantly rebuilt
COPY . .