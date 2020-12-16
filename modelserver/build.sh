sed -i '/visual-attention/d' ./requirements.txt
sed -i '/detectron2/d' ./requirements.txt
docker build -t modelserver:$(git rev-parse --short HEAD) .