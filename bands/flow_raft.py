import os
import sys
sys.path.append('raft')

import json
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
from common.encode import process_flow, encode_flow

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
ITERATIONS = 20
MODEL = 'models/raft-things.pth'
data = None


def load_image(image):
    img = np.array(image)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


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


def run(args):
    # load model
    print("Open RAFT Data pipeline")
    model = torch.nn.DataParallel(RAFT(args))
    print("load model " + MODEL + " on " + DEVICE)
    model.load_state_dict( torch.load( args.model ) )
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    # load video
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()
    
    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output +'.mp4' )

    if args.backwards:
        out_bk_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output +'_bwd.mp4' )

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
    csv_dist = open( args.output +'.csv' , 'w')
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
    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")
    parser.add_argument('-model', '-m', help="model path", type=str, default=MODEL)
    parser.add_argument('-iterations', help="number of iterations", type=int, default=ITERATIONS)
    parser.add_argument('-flo_subpath', '-f', help="path to flo files", type=str, default='')
    parser.add_argument('-ds_subpath', '-d', help="path to flo files", type=str, default='')
    parser.add_argument('-vis_subpath', '-v', help="path to flo files", type=str, default='')
    parser.add_argument('-backwards','-b',  help="Backward video", action='store_true')

    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--raft_model', default='models/raft-things.pth', help="[RAFT] restore checkpoint")
    parser.add_argument('--small', action='store_true', help='[RAFT] use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='[RAFT] use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='[RAFT] use efficent correlation implementation')

    args = parser.parse_args()

    if os.path.isdir( args.input ):
        input_payload = os.path.join( args.input, "payload.json")
        if os.path.isfile(input_payload):
            data = json.load( open(input_payload) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join(input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    if not data:
        data = json.load( open(input_payload) )

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, "flow")
    elif args.output == "":
        args.output = os.path.join(input_folder, "flow")

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
    
    check_overwrite(args.output +'.mp4')
    run(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )

