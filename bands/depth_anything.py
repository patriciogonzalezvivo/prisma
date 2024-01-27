# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#

import numpy as np
import torch
import argparse
import os
import cv2

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from d_anything.dpt import DepthAnything
from d_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from common.encode import heat_to_rgb
from common.io import open_rgb, create_folder, check_overwrite, write_depth, write_pcl
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url

BAND = "depth_anything"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

device = None
model = None
transform = None
data = None

# Load Zoe
def init_model():
    global device, model, transform
    device = torch.device( DEVICE )

    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    # total_params = sum(param.numel() for param in model.parameters())
    # print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    return model


def infer(img, normalize=False):
    global device, model, transform

    if model == None:
        init_model()

    image = img / 255.0

    h, w = image.shape[:2]
        
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = model(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    prediction = depth.cpu().numpy()

    if normalize:
        # Normalization
        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)

    return prediction


def process_image(args):
    # LOAD resource 
    in_image = open_rgb(args.input)
    output_folder = os.path.dirname(args.output)

    prediction = infer( in_image, normalize=False)

    if data:
        depth_min = prediction.min().item()
        depth_max = prediction.max().item()
        data["bands"][BAND]["values"] = { 
                                            "min" : {
                                                    "value": depth_min, 
                                                    "type": "float"
                                            },
                                            "max" : {
                                                "value": depth_max,
                                                "type": "float" 
                                            }
                                        }
    if args.npy:
        np.save( os.path.join(output_folder, BAND + '.npy'), prediction)

    if args.ply:
        write_pcl( os.path.join(output_folder, BAND + '.ply'), prediction, np.array(in_image), flip=True)

    # Save depth
    write_depth( args.output, prediction, normalize=True, heatmap=True, encode_range=True, flip=True)


def process_video(args):
    import decord
    from tqdm import tqdm
    from common.io import VideoWriter

    # LOAD resource 
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()

    # width /= 2
    # height /= 2

    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output)
    output_folder = os.path.dirname(args.output)
    
    if args.subpath != '':
        if data:
            data["bands"][BAND]["folder"] = args.subpath
        args.subpath = os.path.join(output_folder, args.subpath)
        create_folder(args.subpath)

    csv_files = []
    for i in tqdm( range(total_frames) ):
        img = in_video[i].asnumpy()

        prediction = infer( img, normalize=False )

        if args.npy:
            if args.subpath != '':
                np.save( os.path.join(args.subpath, "{:05d}.npy".format(i)), prediction)
            else:
                np.save( os.path.join(os.path.join(output_folder, BAND + '_npy', prediction), '%04d.npy' % i), prediction)

        # Normalize for video encoding (min/max are saved on two CVS files)
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth = (prediction - depth_min) / (depth_max - depth_min)
        depth = 1.0 - depth
        out_video.write( ( heat_to_rgb( depth.astype(np.float64) ) * 255 ).astype(np.uint8) )
        csv_files.append( ( depth_min.item(), depth_max.item()  ) )

        # Safe prediction (not normalized depth, so it can encode the range)
        if args.subpath != '':
            write_depth( os.path.join(args.subpath, "{:05d}.png".format(i)), prediction, normalize=True, flip=True, heatmap=True, encode_range=True)

    # Close Video
    out_video.close()

    # Save min/max depth in CVS files
    output_folder = os.path.dirname(args.output)
    csv_min = open( os.path.join( output_folder, BAND + "_min.csv" ) , 'w')
    csv_max = open( os.path.join( output_folder, BAND + "_max.csv" ) , 'w')
    for e in csv_files:
        csv_min.write( '{}\n'.format(e[0]) )
        csv_max.write( '{}\n'.format(e[1]) )
    csv_min.close()
    csv_max.close()

    # Save metadata
    if data:
        data["bands"][BAND]["values"] = { 
                                            "min" : {
                                                    "type": "float",
                                                    "url": BAND + "_min.csv"
                                            },
                                            "max" : {
                                                "type": "float", 
                                                "url": BAND + "_max.csv",
                                            }
                                        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', help="Input image/video", type=str, required=True)
    parser.add_argument('--output', '-o', help="Output image/video", type=str, default="")
    parser.add_argument('--npy' , '-n', help="Save numpy data", action='store_true')
    parser.add_argument('--ply' , '-p', help="Create point cloud PLY", action='store_true')
    parser.add_argument('--subpath', '-d', help="subpath to frames", type=str, default='')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    args = parser.parse_args()
    
    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.input = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output, force_extension="png")

    # Check if the output folder exists
    check_overwrite(args.output)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # init model
    init_model()

    # compute depth maps on video or image
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    write_metadata(args.input, data)

