#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def runVideo(args):
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
    mask_video.close()
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename
    data["bands"][BAND]["ids"] = CLASSES

    # SDF
    if args.sdf:
        sdf_video.close()
        data["bands"][BAND + "_sdf"] = { }
        data["bands"][BAND + "_sdf"]["url"] = sdf_filename


def runImage(args):
    global model, device

    # Export properties
    output_folder = os.path.dirname(args.output)
    output_filename = os.path.basename(args.output)
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

    parser.add_argument('--input', '-i', help="input", type=str, required=True)
    parser.add_argument('--output', '-o', help="output", type=str, default="")

    parser.add_argument("--sdf", action='store_true')

    parser.add_argument('--mask_subpath', '-md', help="Mask Subpath to frames", type=str, default='')
    parser.add_argument('--sdf_subpath', '-sd', help="SDF Subpath to frames", type=str, default='')

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
        runVideo(args)
    else:
        runImage(args)

    # save metadata
    if data:
        write_metadata(args.input, data)
