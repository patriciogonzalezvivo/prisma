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
import cv2

import torch

import warnings
warnings.filterwarnings("ignore")

import decord
from tqdm import tqdm

from gmflow.gmflow import GMFlow
import torch.nn.functional as F

from common.io import VideoWriter, check_overwrite
from common.meta import load_metadata, get_target, write_metadata, get_url
from common.flow import load_image, InputPadder, compute_fwdbwd_mask, write_flow


BAND = "flow_gmflow"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
ITERATIONS = 20
MODEL = 'models/gmflow_sintel-0c07dcb3.pth'

data = None
device = None
model = None
optimizer = None

def init_model(args):
    global device, model, optimizer 
    
    device = torch.device(DEVICE)
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    print('Load checkpoint: %s' % args.model)
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(args.model, map_location=loc)
    
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=args.strict_resume)

    return model


def infer(args, image1, image2):
    fwd_flow, bwd_flow = None, None
    fwd_mask, bwd_mask = None, None

    if args.inference_size is None:
        padder = InputPadder(image1.shape, padding_factor=args.padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
    else:
        image1, image2 = image1[None].cuda(), image2[None].cuda()

    if args.inference_size is not None:
        assert isinstance(args.inference_size, list) or isinstance(args.inference_size, tuple)
        ori_size = image1.shape[-2:]
        image1 = F.interpolate(image1, size=args.inference_size, mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=args.inference_size, mode='bilinear', align_corners=True)

    results_dict = model(image1, image2,
                            attn_splits_list=args.attn_splits_list,
                            corr_radius_list=args.corr_radius_list,
                            prop_radius_list=args.prop_radius_list,
                            pred_bidir_flow=args.output_mask != '' or args.subpath_mask != '' or args.backwards,
                        )
    
    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

    # resize back
    if args.inference_size is not None:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / args.inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / args.inference_size[-2]

    if args.inference_size is None:
        fwd_flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
    else:
        fwd_flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    # also predict backward flow
    if args.output_mask != '' or args.subpath_mask != '' or args.backwards:
        assert flow_pr.size(0) == 2  # [2, H, W, 2]

        if args.inference_size is None:
            bwd_flow = padder.unpad(flow_pr[1]).permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        else:
            bwd_flow = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

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
    
    fwd_flow_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )
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

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)

        curr_frame = load_image( frame )

        if prev_frame is not None:
            with torch.no_grad():
                fwd_flow, bwd_flow, fwd_mask, bwd_mask = infer(args, prev_frame, curr_frame)
                write_flow(args, fwd_flow, fwd_flow_video, max_disps, i-1, 
                            fwd_mask=fwd_mask, fwd_mask_video=fwd_mask_video,
                            bwd_flow=bwd_flow, bwd_flow_video=bwd_flow_video, 
                            bwd_mask=bwd_mask, bwd_mask_video=bwd_mask_video)
                
        prev_frame = curr_frame.clone()

    # Save last frame
    fwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
    bwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)

    if args.subpath_mask != '':
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

    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--model', '-m', help="model path", type=str, default=MODEL)

    # GMFlow model
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--num_scales', default=1, type=int, help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+', help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+', help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+', help='self-attention radius for flow propagation, -1 indicates global attention')

    # resume pretrained model or resume training
    parser.add_argument('--strict_resume', action='store_true')

    # inference on a directory
    parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='can specify the inference size')
    parser.add_argument('--padding_factor', default=16, type=int, help='the input should be divisible by padding_factor, otherwise do padding')
    
    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)

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

