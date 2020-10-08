#!/bin/bash
docker run --rm --gpus all -d -ti -p "8888:8888" --network="captivate_docker_default" --mount type=bind,source="$(pwd)",target=/workspace/modelserver \
        modelserver:$(git rev-parse --short HEAD) \
        bash
