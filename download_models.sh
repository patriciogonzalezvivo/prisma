#!/bin/bash

# RAFTs
# wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
wget https://www.dropbox.com/scl/fi/zh3220fwav2l8i2zbvreo/models.zip?rlkey=dg8bynghzm1xaeqkmnvlv1q3n&dl=0 -O models.zip
unzip models.zip
rm models.zip

cd models

# MMDetect
wget https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r101_fpn_3x_coco/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r101_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_3x_coco.py
wget https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/solov2/solov2_r50_fpn_1x_coco.py

# DepthAnything

wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt?download=true
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt?download=true
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true

cd ..