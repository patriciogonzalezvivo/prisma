#!/bin/bash

# RAFTs
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
rm models.zip

cd models

# MIDAS
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt

# MMDetect
wget https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r101_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_1x_coco.py

# Perspective Fields
wget https://raw.githubusercontent.com/jinlinyi/PerspectiveFields/main/jupyter-notebooks/models/cvpr2023.yaml

# FcF-Inpainting
wget https://shi-labs.com/projects/fcf-inpainting/places_512.pkl
# wget https://shi-labs.com/projects/fcf-inpainting/places.pkl
# wget https://shi-labs.com/projects/fcf-inpainting/celeba-hq.pkl

# RealESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

cd ..