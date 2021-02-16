# FROM python:3.8.5-buster
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# make sure this matches the bind mount volume in docker-compose!!!
WORKDIR /workspace/modelserver

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    gcc musl-dev ffmpeg \
    g++ default-jdk git make


COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# install local packages
COPY ./models ./models
# RUN pip install ./models/retinaface
RUN pip install ./models/visual-attention
# RUN pip install -e ./models/arcface

# install detectron2
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

######
###### install Khaiii
WORKDIR /workspace
RUN git clone https://github.com/kakao/khaiii.git

WORKDIR /workspace/khaiii
RUN mkdir build
WORKDIR /workspace/khaiii/build
RUN cmake ..

RUN make all

RUN make resource
RUN make install
RUN make package_python

WORKDIR /workspace/khaiii/build/package_python
RUN pip install .

RUN apt-get update -y
RUN apt-get install -y language-pack-ko
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
#####

# RUN python ./models/yolact/external/DCNv2/setup.py build develop

# still necessary to do this besides bind mount so that files are persisted
# this should be the last layer so that deeper layers aren't constantly rebuilt
WORKDIR /workspace/modelserver
COPY . .
