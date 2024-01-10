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


def process_video(args):
    global model
    
    # load video
    output_basename = args.output.rsplit( ".", 1 )[ 0 ]
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()
    
    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )

    if args.backwards:
        out_bk_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_basename +'_bwd.mp4' )

    max_disps = []

    prev_frame = None
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

                if args.ds_subpath != '' or args.vis_subpath  != '' or args.flo_subpath != '' or args.backwards:
                    bwd_flow = padder.unpad(flow_up[1]).permute(1,2,0).cpu().numpy()

                if args.ds_subpath != '':
                    mask_fwd, mask_bwd = compute_fwdbwd_mask(fwd_flow, bwd_flow)
        else:
            fwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)
            bwd_flow = np.zeros(frame[..., :2].shape, dtype=np.float32)

            if args.ds_subpath != '':
                mask_fwd = np.zeros(frame[..., 0].shape, dtype=bool)
                mask_bwd = np.zeros(frame[..., 0].shape, dtype=bool)

        flow_pixels, max_disp = process_flow(fwd_flow)
        out_video.write(flow_pixels)
        max_disps.append(max_disp)

        if args.backwards:        
            bwd_flow_pixels, _, _ = process_flow(bwd_flow)
            out_bk_video.write(bwd_flow_pixels)

        if args.flo_subpath != '':
            write_flow(os.path.join(args.flo_subpath + '_fwd', '%04d.flo' % i), fwd_flow)
            write_flow(os.path.join(args.flo_subpath + '_bwd', '%04d.flo' % i), bwd_flow)

        if args.ds_subpath != '':
            cv2.imwrite(os.path.join(args.ds_subpath + '_fwd', '%04d.png' % i), encode_flow(fwd_flow, mask_fwd))
            cv2.imwrite(os.path.join(args.ds_subpath + '_bwd', '%04d.png' % i), encode_flow(bwd_flow, mask_bwd))

        if args.vis_subpath != '':
            cv2.imwrite(os.path.join(args.vis_subpath + '_fwd', '%04d.jpg' % i), flow_viz.flow_to_image(fwd_flow))
            cv2.imwrite(os.path.join(args.vis_subpath + '_bwd', '%04d.jpg' % i), flow_viz.flow_to_image(bwd_flow))

        prev_frame = curr_frame.clone()
    
    # Save and close video
    out_video.close()
    if args.backwards:
        out_bk_video.close()

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
    parser.add_argument('--flo_subpath', '-f', help="path to flo files", type=str, default='')
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

    if args.flo_subpath != '':
        args.flo_subpath = os.path.join(input_folder, args.flo_subpath)
        os.makedirs(args.flo_subpath + "_fwd", exist_ok=True)
        os.makedirs(args.flo_subpath + "_bwd", exist_ok=True)

    if args.ds_subpath != '':
        args.ds_subpath = os.path.join(input_folder, args.ds_subpath)
        os.makedirs(args.ds_subpath + "_fwd", exist_ok=True)
        os.makedirs(args.ds_subpath + "_bwd", exist_ok=True)

    if args.vis_subpath != '':
        args.vis_subpath = os.path.join(input_folder, args.vis_subpath)
        os.makedirs(args.vis_subpath + "_fwd", exist_ok=True)
        os.makedirs(args.vis_subpath + "_bwd", exist_ok=True)

    # init model
    init_model(args)
    
    # compute optical flow
    process_video(args)

    # save metadata
    write_metadata(args.input, data)

