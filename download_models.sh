#!/bin/bash

# RAFTs
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip

cd models

# MMDetect
wget https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r101_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_1x_coco.py

# FcF-Inpainting
wget https://shi-labs.com/projects/fcf-inpainting/places_512.pkl
# wget https://shi-labs.com/projects/fcf-inpainting/places.pkl
# wget https://shi-labs.com/projects/fcf-inpainting/celeba-hq.pkl

cd ..