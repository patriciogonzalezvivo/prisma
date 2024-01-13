# Copyright (c) 2024, Patricio Gonzalez Vivo
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

#     * Neither the name of Patricio Gonzalez Vivo nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import argparse
import torch
import os

import warnings
warnings.filterwarnings("ignore")

import cv2
from mmdet.apis import init_detector, inference_detector

import snowy

from common.io import create_folder, check_overwrite
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import hue_to_rgb

BAND = "mask"

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
WIDTH = int(1280)
HEIGHT = int(720)
CONFIG = 'models/solov2_r101_fpn_3x_coco.py'
MODEL  = 'models/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth'
# CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
CONFIDENCE_THRESHOLD = 0.7


data = None
model = None
device = None

def init_model():
    global model, device
    device = torch.device( DEVICE )
    model = init_detector(CONFIG, MODEL, device=device)


def getConfidence(result, category, index):
    return result[0][category][index][4]


def getTotalMasks(result, category=0, conf=CONFIDENCE_THRESHOLD):
    total = 0;
    for mask in result[0][category]:
        if mask[4] > conf:
            total += 1
    return total


def getMaskConfidence(result, category, index):
    return result[0][category][index][4]


def getMaskRGB(result, category, index):
    masks = [np.where(m == 1, 255, m) for m in result[1][category][index]]
    return np.stack([masks] * 3, axis=-1)


def getSDF(masks):
    mask = snowy.rgb_to_luminance( snowy.extract_rgb(masks) )
    sdf = snowy.generate_sdf(mask != 0.0)
    sdf = (sdf + 127.0) / 255.0
    sdf = (sdf - 0.25) * 2.0
    return 1.0-np.clip(sdf, 0.0, 1.0)


def process_image(args):
    global model, device

    # Export properties
    output_filename = os.path.basename(args.output)    
    img = cv2.imread(args.input)

    # run the inferance
    result = inference_detector(model, img)
    masks = np.zeros(img.shape)

    total_categories = len(result[0])
    for c in range(total_categories):
        category = model.CLASSES[c]
        total_masks = getTotalMasks(result, c)
        if category in CLASSES:
            for i in range(total_masks):
                if getConfidence(result, c, i) > args.confidence:
                    mask = getMaskRGB(result, c, i)
                    masks = masks + mask
        
    # SDF
    if args.sdf:
        sdf = getSDF(masks)
        masks[..., 1] = sdf[...,0] * 255

    # Mask
    cv2.imwrite(args.output, masks.astype(np.uint8))
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename
    data["bands"][BAND]["ids"] = CLASSES


def process_video(args):
    global model, device

    import decord
    from tqdm import tqdm
    from common.io import VideoWriter

    # LOAD resource 
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()

    width /= 2
    height /= 2

    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
        
    mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

    if args.subpath != '':
        if data:
            data["bands"][BAND]["folder"] = args.subpath
        args.subpath = os.path.join(output_folder, args.subpath)
        create_folder(args.subpath)

    for f in tqdm( range(total_frames) ):
        img = cv2.cvtColor(in_video[f].asnumpy(), cv2.COLOR_BGR2RGB)
        result = inference_detector(model, img)
        masks = np.zeros(img.shape)
        total_categories = len(result[0])
        for c in range(total_categories):
            category = model.CLASSES[c]
            total_masks = getTotalMasks(result, c)
            if category in CLASSES:
                for i in range(total_masks):
                    if getConfidence(result, c, i) > args.confidence:
                        mask = getMaskRGB(result, c, i)
                        masks = masks + mask
        
        if args.sdf:
            sdf = getSDF(masks)
            masks[..., 1] = sdf[...,0] * 255

        mask_video.write( masks.astype(np.uint8) )

        if args.subpath != '':
            cv2.imwrite(os.path.join(args.subpath, "{:05d}.png".format(f)), masks.astype(np.uint8))

    # Mask
    mask_video.close()
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename
    data["bands"][BAND]["ids"] = CLASSES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', help="input", type=str, required=True)
    parser.add_argument('--output', '-o', help="output", type=str, default="")
    parser.add_argument('--confidence', '-c', help="confidence threshold", type=float, default=CONFIDENCE_THRESHOLD)

    parser.add_argument('--sdf', '-s',action='store_true')
    parser.add_argument('--subpath', help="Mask Subpath to frames", type=str, default='')

    args = parser.parse_args()

    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.input = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output, force_image_extension="png")

    # Check if the output folder exists
    check_overwrite(args.output)

    # Load model
    init_model()

    # compute depth maps
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    if data:
        write_metadata(args.input, data)
