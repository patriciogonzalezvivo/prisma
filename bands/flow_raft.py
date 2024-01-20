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
from raft.utils import flow_viz
from raft.utils.utils import InputPadder
from raft.utils.frame_utils import write_flow

from common.io import VideoWriter, check_overwrite
from common.meta import load_metadata, get_target, write_metadata, get_url
from common.encode import process_flow, encode_flow

BAND = "flow"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
ITERATIONS = 20
MODEL = 'models/raft-things.pth'


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


def load_image(image):
    img = np.array(image)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow, alpha_1=0.05, alpha_2=0.5):
    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask


def write_flow(args, fwd_flow, fwd_flow_video, max_disps, idx, bwd_flow=None, bwd_flow_video=None, fwd_mask=None, bwd_mask=None):
    flow_pixels, max_disp = process_flow(fwd_flow)
    fwd_flow_video.write(flow_pixels)
    max_disps.append(max_disp)

    if args.backwards and bwd_flow_video:
        bwd_flow_pixels, _, _ = process_flow(bwd_flow)
        bwd_flow_video.write(bwd_flow_pixels)

    if args.subpath != '':
        write_flow(os.path.join(args.subpath + '_fwd', '%04d.flo' % idx), fwd_flow)
        if args.backwards:
            write_flow(os.path.join(args.subpath + '_bwd', '%04d.flo' % idx), bwd_flow)

    if args.ds_subpath != '':
        cv2.imwrite(os.path.join(args.ds_subpath + '_fwd', '%04d.png' % idx), encode_flow(fwd_flow, fwd_mask))
        if args.backwards:
            cv2.imwrite(os.path.join(args.ds_subpath + '_bwd', '%04d.png' % idx), encode_flow(bwd_flow, bwd_mask))

    if args.vis_subpath != '':
        cv2.imwrite(os.path.join(args.vis_subpath + '_fwd', '%04d.jpg' % idx), flow_viz.flow_to_image(fwd_flow))
        if args.backwards:
            cv2.imwrite(os.path.join(args.vis_subpath + '_bwd', '%04d.jpg' % idx), flow_viz.flow_to_image(bwd_flow))


def process_video(args):
    global model
    
    # load video
    output_basename = args.output.rsplit( ".", 1 )[ 0 ]
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()
    
    fwd_flow_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )

    bwd_flow_video = None
    if args.backwards:
        bwd_flow_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_basename +'_bwd.mp4' )

    max_disps = []

    prev_frame = None
    bwd_flow = None
    fwd_mask = None
    bwd_mask = None
    for i in tqdm( range(total_frames) ):
        frame = in_video[i].asnumpy()
        ds_frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        curr_frame = load_image( ds_frame )

        if prev_frame is not None:
            with torch.no_grad():
                image1 = torch.cat([prev_frame, curr_frame], dim=0)
                image2 = torch.cat([curr_frame, prev_frame], dim=0)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = model(image1, image2, iters=args.iterations, test_mode=True)
                fwd_flow = padder.unpad(flow_up[0]).permute(1,2,0).cpu().numpy()

                if args.ds_subpath != '' or args.vis_subpath  != '' or args.subpath != '' or args.backwards:
                    bwd_flow = padder.unpad(flow_up[1]).permute(1,2,0).cpu().numpy()

                if args.ds_subpath != '':
                    fwd_mask, bwd_mask = compute_fwdbwd_mask(fwd_flow, bwd_flow)

            write_flow(args, fwd_flow, fwd_flow_video, max_disps, i, bwd_flow=bwd_flow, bwd_flow_video=bwd_flow_video, fwd_mask=fwd_mask, bwd_mask=bwd_mask)

        prev_frame = curr_frame.clone()

    fwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
    bwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)

    if args.ds_subpath != '':
        fwd_mask = np.zeros(frame[..., 0].shape, dtype=bool)
        bwd_mask = np.zeros(frame[..., 0].shape, dtype=bool)
    
    write_flow(args, fwd_flow, fwd_flow_video, max_disps, i, bwd_flow=bwd_flow, bwd_flow_video=bwd_flow_video, fwd_mask=fwd_mask, bwd_mask=bwd_mask)
    
    # Save and close video
    fwd_flow_video.close()
    if args.backwards:
        bwd_flow_video.close()

    # Save max disatances per frame as a CSV file
    csv_dist = open( output_basename +'.csv' , 'w')
    for e in max_disps:
        csv_dist.write( '{}\n'.format(e) )
    csv_dist.close()

    if data:
        data["bands"]["flow"] = { 
            "url": "flow.mp4",
            "values": {
                "dist" : {
                    "type": "float",
                    "url": "flow.csv"
                }
            }
        }

        if args.backwards:
            data["bands"]["flow_bwd"] = { 
                "url": "flow_bwd.mp4",
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input", type=str, required=True)
    parser.add_argument('--output', '-o', help="output", type=str, default="")
    parser.add_argument('--model', '-m', help="model path", type=str, default=MODEL)
    parser.add_argument('--iterations', help="number of iterations", type=int, default=ITERATIONS)
    parser.add_argument('--subpath', '-f', help="path to flo files", type=str, default='')
    parser.add_argument('--ds_subpath', '-d', help="path to flo files", type=str, default='')
    parser.add_argument('--vis_subpath', '-v', help="path to flo files", type=str, default='')
    parser.add_argument('--backwards','-b',  help="Backward video", action='store_true')

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

    # Check if the output folder exists
    check_overwrite(args.output)

    input_folder = os.path.dirname(args.input)

    if args.subpath != '':
        args.subpath = os.path.join(input_folder, args.subpath)
        os.makedirs(args.subpath + "_fwd", exist_ok=True)
        if args.backwards:
            os.makedirs(args.subpath + "_bwd", exist_ok=True)

    if args.ds_subpath != '':
        args.ds_subpath = os.path.join(input_folder, args.ds_subpath)
        os.makedirs(args.ds_subpath + "_fwd", exist_ok=True)
        if args.backwards:
            os.makedirs(args.ds_subpath + "_bwd", exist_ok=True)

    if args.vis_subpath != '':
        args.vis_subpath = os.path.join(input_folder, args.vis_subpath)
        os.makedirs(args.vis_subpath + "_fwd", exist_ok=True)
        if args.backwards:
            os.makedirs(args.vis_subpath + "_bwd", exist_ok=True)

    # init model
    init_model(args)
    
    # compute optical flow
    process_video(args)

    # save metadata
    write_metadata(args.input, data)

