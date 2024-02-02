# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#

import os
import sys
sys.path.append('raft')

import argparse

import numpy as np
import torch

import cv2

import warnings
warnings.filterwarnings("ignore")

import decord
from tqdm import tqdm

from raft.raft import RAFT

from common.io import VideoWriter, check_overwrite
from common.meta import load_metadata, get_target, write_metadata, get_url
from common.flow import load_image, compute_fwdbwd_mask, InputPadder, write_flow

BAND = "flow_raft"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
ITERATIONS = 20
MODEL = 'models/raft-sintel.pth'

data = None
model = None

def init_model(args):
    global model, device
    
    # load model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict( torch.load( args.model ) )
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    return model


def infer(args, image1, image2):
    fwd_flow, bwd_flow = None, None
    fwd_mask, bwd_mask = None, None

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    _, flow_up = model(image1, image2, iters=args.iterations, test_mode=True)
    fwd_flow = padder.unpad(flow_up[0]).permute(1,2,0).cpu().numpy()

    if args.output_mask != '' or args.subpath_mask != '' or args.subpath != '' or args.backwards:
        bwd_flow = padder.unpad(flow_up[1]).permute(1,2,0).cpu().numpy()

    if args.output_mask != '' or args.subpath_mask != '':
        fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, bwd_flow)

    return fwd_flow, bwd_flow, fwd_mask, bwd_mask


def process_video(args):
    global model
    
    # load video
    output_basename = args.output.rsplit( ".", 1 )[ 0 ]
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()
    
    fwd_flow_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output)
    fwd_mask_video = None
    if args.output_mask != '':
        fwd_mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output_mask)

    bwd_flow_video = None
    bwd_mask_video = None
    if args.backwards:
        bwd_flow_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_basename +'_bwd.mp4' )
        if args.output_mask != '':
            bwd_mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output_mask.rsplit( ".", 1 )[ 0 ] +'_bwd.mp4')

    max_disps = []

    prev_frame = None
    bwd_flow = None
    fwd_mask = None
    bwd_mask = None
    for i in tqdm( range(total_frames) ):
        frame = in_video[i].asnumpy()
        ds_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        curr_frame = load_image( ds_frame )[None].to(DEVICE)

        if prev_frame is not None:
            with torch.no_grad():
                image1 = torch.cat([prev_frame, curr_frame], dim=0)
                image2 = torch.cat([curr_frame, prev_frame], dim=0)
                fwd_flow, bwd_flow, fwd_mask, bwd_mask = infer(args, image1, image2)
                write_flow(args, fwd_flow, fwd_flow_video, max_disps, i-1, 
                            fwd_mask=fwd_mask, fwd_mask_video=fwd_mask_video,
                            bwd_flow=bwd_flow, bwd_flow_video=bwd_flow_video, 
                            bwd_mask=bwd_mask, bwd_mask_video=bwd_mask_video)

        prev_frame = curr_frame.clone()

    # Last frame
    fwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
    bwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)

    if args.output_mask != '' or args.subpath_mask != '':
        fwd_mask = np.zeros(frame[..., 0].shape, dtype=bool)
        bwd_mask = np.zeros(frame[..., 0].shape, dtype=bool)
    
    write_flow(args, fwd_flow, fwd_flow_video, max_disps, i, 
                fwd_mask=fwd_mask, fwd_mask_video=fwd_mask_video,
                bwd_flow=bwd_flow, bwd_flow_video=bwd_flow_video, 
                bwd_mask=bwd_mask, bwd_mask_video=bwd_mask_video)
    
    # Save and close video
    fwd_flow_video.close()
    if fwd_mask_video:
        fwd_mask_video.close()
    if bwd_flow_video:
        bwd_flow_video.close()
    if bwd_mask_video:
        bwd_mask_video.close()

    # Save max disatances per frame as a CSV file
    csv_dist = open( output_basename +'.csv' , 'w')
    for e in max_disps:
        csv_dist.write( '{}\n'.format(e) )
    csv_dist.close()

    if data:
        data["bands"][BAND] = { 
            "url": BAND + ".mp4",
            "values": {
                "dist" : {
                    "type": "float",
                    "url": BAND + ".csv"
                }
            }
        }

        if args.subpath != '':
            data["bands"][BAND]["folder"] = args.subpath

        if args.backwards:
            data["bands"][BAND + "_bwd"] = { "url": BAND + "_bwd.mp4", }
            if args.subpath != '':
                data["bands"][BAND + "_bwd"]["folder"] = args.subpath + "_bwd"

        if args.output_mask != '':
            data["bands"][BAND + "_mask"] = { "url": BAND + "_mask.mp4", }

            if args.backwards:
                data["bands"][BAND + "_mask_bwd"] = { "url": BAND + "_mask_bwd.mp4", }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input", type=str, required=True)

    parser.add_argument('--output', '-o', help="output", type=str, default="")
    parser.add_argument('--subpath', help="path to flo files", type=str, default="")
    parser.add_argument('--backwards','-b',  help="Backward video", action='store_true')
    parser.add_argument('--mask', action='store_true', help="Compute mask as well")
    parser.add_argument('--output_mask', help="output dense", type=str, default="")
    parser.add_argument('--subpath_mask', help="path to flo files", type=str, default="")

    parser.add_argument('--iterations', help="number of iterations", type=int, default=ITERATIONS)
    parser.add_argument('--model', '-m', help="model path", type=str, default=MODEL)

    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--raft_model', default='models/raft-things.pth', help="[RAFT] restore checkpoint")
    parser.add_argument('--small', action='store_true', help='[RAFT] use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='[RAFT] use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='[RAFT] use efficent correlation implementation')

    args = parser.parse_args()

    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.input = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output)
        if args.mask:
            args.output_mask = get_target(args.input, data, band=(BAND + "_mask"))

    # Check if the output folder exists
    check_overwrite(args.output)

    input_folder = os.path.dirname(args.input)

    if args.subpath != '':
        args.subpath = os.path.join(input_folder, args.subpath)
        os.makedirs(args.subpath + "_fwd", exist_ok=True)
        if args.backwards:
            os.makedirs(args.subpath + "_bwd", exist_ok=True)

    if args.subpath_mask != '':
        args.subpath_mask = os.path.join(input_folder, args.subpath_mask)
        os.makedirs(args.subpath_mask + "_fwd", exist_ok=True)
        if args.backwards:
            os.makedirs(args.subpath_mask + "_bwd", exist_ok=True)

    # init model
    init_model(args)
    
    # compute optical flow
    process_video(args)

    # save metadata
    write_metadata(args.input, data)

