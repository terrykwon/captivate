#! /bin/bash

VBR="2500k"
FPS="15"
QUAL="medium"
URL="rtmp://localhost:1935/live/test"
SOURCE="/data/pci/video/test_video_9.mkv"

ffmpeg \
    -re -i "$SOURCE" \
    -vcodec libx264 -pix_fmt yuv420p -preset $QUAL -r $FPS -g $(($FPS * 2)) -b:v $VBR \
    -acodec libmp3lame -ar 44100 -threads 6 -qscale 3 -b:a 712000 -bufsize 512k \
    -f flv "$URL"
