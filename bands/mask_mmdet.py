#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import torch
import json
import os

import warnings
warnings.filterwarnings("ignore")

import cv2
from mmdet.apis import init_detector, inference_detector

import snowy
from common.encode import hue_to_rgb
from common.io import create_folder, check_overwrite

BAND = "mask"

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
WIDTH = int(1280)
HEIGHT = int(720)
CONFIG = 'models/solov2_r101_fpn_3x_coco.py'
MODEL  = 'models/solov2_r101_fpn_3x_coco_20220511_095119-c559a076.pth'
# CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASSES = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

data = None
model = None
device = None

def init_model():
    global model, device
    device = torch.device( DEVICE )
    model = init_detector(CONFIG, MODEL, device=device)


def getTotalMasks(result, category=0, conf=0.2):
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


def runVideo(args, data = None):
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
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    if args.mask_subpath != '':
        args.mask = True

    if args.mask:
        mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

        if args.mask_subpath != '':
            if data:
                data["bands"][BAND]["folder"] = args.mask_subpath
            args.mask_subpath = os.path.join(output_folder, args.mask_subpath)
            create_folder(args.mask_subpath)

    if args.sdf_subpath != '':
        args.sdf = True

    if args.sdf:
        sdf_filename = output_basename + "_sdf." + output_extension
        sdf_path = output_folder + "/" + sdf_filename
        sdf_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=sdf_path )

        if args.sdf_subpath != '':
            if data:
                data["bands"][BAND + "_sdf"]["folder"] = args.sdf_subpath
            args.sdf_subpath = os.path.join(output_folder, args.sdf_subpath)
            create_folder(args.sdf_subpath)

    for f in tqdm( range(total_frames) ):
        img = cv2.cvtColor(in_video[f].asnumpy(), cv2.COLOR_BGR2RGB)
        result = inference_detector(model, img)
        masks = np.zeros(img.shape)
        total_categories = len(result[0])
        for c in range(total_categories):
            category = model.CLASSES[c]
            total_masks = getTotalMasks(result, c)
            color = hue_to_rgb( float(c) / float(total_categories) )
            for i in range(total_masks):
                mask = getMaskRGB(result, c, i)
                if category in CLASSES:
                    masks = masks + mask

        if args.mask:
            mask_video.write( masks.astype(np.uint8) )

            if args.mask_subpath != '':
                cv2.imwrite(os.path.join(args.mask_subpath, "{:05d}.png".format(f)), masks.astype(np.uint8))

        if args.sdf:
            masks = snowy.rgb_to_luminance( snowy.extract_rgb(masks) )
            sdf = snowy.generate_sdf(masks != 0.0)
            sdf = (sdf + 127.0) / 255.0
            sdf = np.clip((sdf - 0.25) * 2.0, 0.0, 1.0)
            sdf *= 255
            sdf = cv2.merge((sdf,sdf,sdf))
            sdf_video.write( sdf.astype(np.uint8) )

            if args.sdf_subpath != '':
                cv2.imwrite(os.path.join(args.sdf_subpath, "{:05d}.png".format(f)), sdf.astype(np.uint8))

    # Mask
    if args.mask:
        mask_video.close()
        data["bands"][BAND] = { }
        data["bands"][BAND]["url"] = output_filename
        data["bands"][BAND]["ids"] = CLASSES

    # SDF
    if args.sdf:
        sdf_video.close()
        data["bands"][BAND + "_sdf"] = { }
        data["bands"][BAND + "_sdf"]["url"] = sdf_filename


def runImage(args, data = None):
    global model, device

    # Export properties
    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    sdf_filename = output_basename + "_sdf." + output_extension
    sdf_path = output_folder + "/" + sdf_filename
    
    img = cv2.imread(args.input)

    # run the inferance
    result = inference_detector(model, img)
    masks = np.zeros(img.shape)

    total_categories = len(result[0])
    for c in range(total_categories):
        category = model.CLASSES[c]
        total_masks = getTotalMasks(result, c)
        
        for i in range(total_masks):
            mask = getMaskRGB(result, c, i)
            if category in CLASSES:
                masks = masks + mask
        
    # Mask
    if args.mask:
        cv2.imwrite(args.output, masks.astype(np.uint8))
        data["bands"][BAND] = { }
        data["bands"][BAND]["url"] = output_filename
        data["bands"][BAND]["ids"] = CLASSES

    # SDF
    if args.sdf:
        mask = snowy.rgb_to_luminance( snowy.extract_rgb(masks) )
        sdf = snowy.generate_sdf(mask != 0.0)
        sdf = (sdf + 127.0) / 255.0
        sdf = (sdf - 0.25) * 2.0
        snowy.export(sdf, sdf_path)
        data["bands"][BAND + "_sdf"] = { }
        data["bands"][BAND + "_sdf"]["url"] = sdf_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")

    parser.add_argument("-mask", action='store_true')
    parser.add_argument("-sdf", action='store_true')

    parser.add_argument('-mask_subpath', '-md', help="Mask Subpath to frames", type=str, default='')
    parser.add_argument('-sdf_subpath', '-sd', help="SDF Subpath to frames", type=str, default='')

    args = parser.parse_args()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join( input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit(".", 1)[0]
    input_extension = input_filename.rsplit(".", 1)[1]
    input_video = input_extension == "mp4"

    if not input_video:
        input_extension = "png"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + "." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, BAND + "." + input_extension)
        
    check_overwrite(args.output)

    init_model()

    if os.path.isfile(input_payload):
        if not data:
            data = json.load( open(input_payload) )

    # compute depth maps
    if input_video:
        runVideo(args, data)
    else:
        runImage(args, data)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )
