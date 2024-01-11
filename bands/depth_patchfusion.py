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
import torch
import argparse
import os

import cv2
from tqdm import tqdm

from patchfusion.zoedepth.utils.config import get_config_user
from patchfusion.zoedepth.models.builder import build_model
from patchfusion.zoedepth.utils.arg_utils import parse_unknown
from patchfusion.zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose

from patchfusion.infer_user import load_ckpt
from patchfusion.infer_user import regular_tile, random_tile

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from common.io import open_image, to_float_rgb, check_overwrite, create_folder, write_depth, write_pcl
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import heat_to_rgb

BAND = "depth_patchfusion"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
MODEL = "models/patchfusion_u4k.pt"
# MODEL_TYPE = "zoedepth"
MODEL_TYPE = "zoedepth_custom"
CONFIG = "bands/patchfusion/zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json"

WIDTH = int(1280)
HEIGHT = int(720)

device = None
model = None
data = None
transform = None

# Load Zoe
def init_model():
    global model, device, transform
    device = torch.device( DEVICE )

    transform = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])

    overwrite_kwargs = parse_unknown({})
    overwrite_kwargs['model_cfg_path'] = CONFIG

    config = get_config_user("zoedepth_custom", **overwrite_kwargs)
    config["pretrained_resource"] = ''
    model = build_model(config)
    model = load_ckpt(model, MODEL)
    model.eval()
    model.cuda()
    # model.to(device)
    
    return model


def infer(img, mode="r128", blr_mask=True, boundary=0, interpolation='bicubic', optimize=True, normalize=True):
    global model, transform


    if model == None:
        init_model()

    img_float = to_float_rgb( img )
    img_resolution = (img_float.shape[1], img_float.shape[0])

    RESOLUTION = (2160, 3840)

    # try to reduce resolution
    if img_float.shape[0] <= 480 and img_float.shape[1] <= 640:
        RESOLUTION = (480, 640)
    elif img_float.shape[0] <= 1080 and img_float.shape[1] <= 1920:
        RESOLUTION = (1080, 1920)

    crop = (int(RESOLUTION[0] // 4), int(RESOLUTION[1] // 4))

    img_t = F.interpolate(torch.tensor(img_float).unsqueeze(dim=0).permute(0, 3, 1, 2), RESOLUTION, mode=interpolation, align_corners=True)
    img_t = img_t.squeeze().permute(1, 2, 0)

    img_t = torch.tensor(img_t).unsqueeze(dim=0).permute(0, 3, 1, 2) # shape: 1, 3, h, w
    img_lr = transform(img_t)

    avg_depth_map = regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=0, offset_y=0, img_lr=img_lr)

    if mode== 'p16':
        pass
    elif mode== 'p49':
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=crop[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=0, offset_y=crop[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=crop[1]//2, offset_y=crop[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

    elif mode[0] == 'r':
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=crop[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=0, offset_y=crop[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, offset_x=crop[1]//2, offset_y=crop[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

        for i in tqdm(range(int(mode[1:]))):
            random_tile(model, img_t, RESOLUTION, crop_size=crop, transform=transform, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

    depth = avg_depth_map.average_map.detach().cpu().numpy()
    return cv2.resize(depth, img_resolution, interpolation=cv2.INTER_LINEAR)


def process_image(args):
    output_folder = os.path.dirname(args.output)

    # LOAD resource 
    in_image = open_image(args.input)

    prediction = infer(in_image, mode=args.mode, blr_mask=not args.no_blur, boundary=args.boundary )

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
        write_pcl( os.path.join(output_folder, BAND + '.ply'), prediction, np.array(in_image))

    # Save depth
    write_depth( args.output, prediction, normalize=True, flip=False, heatmap=True, encode_range=True)


def process_video(args):
    import decord
    from common.io import VideoWriter

    # LOAD resource 
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()

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

        prediction = infer( img, mode=args.mode, blr_mask=not args.no_blur, boundary=args.boundary)

        if args.npy:
            if args.subpath != '':
                np.save( os.path.join(args.subpath, "{:05d}.npy".format(i)), prediction)
            else:
                np.save(os.path.join(os.path.join(output_folder, BAND + '_npy', prediction), '%04d.npy' % i), prediction)

        depth_min = prediction.min()
        depth_max = prediction.max()
        depth = (prediction - depth_min) / (depth_max - depth_min)
        out_video.write( ( heat_to_rgb( depth.astype(np.float64) ) * 255 ).astype(np.uint8) )

        if args.subpath != '':
            write_depth( os.path.join(args.subpath, "{:05d}.png".format(i)), prediction, normalize=True, flip=False, heatmap=True, encode_range=True)

        csv_files.append( ( depth_min.item(),
                            depth_max.item()  ) )
    out_video.close()

    csv_min = open( os.path.join( output_folder, BAND + "_min.csv" ) , 'w')
    csv_max = open( os.path.join( output_folder, BAND + "_max.csv" ) , 'w')

    for e in csv_files:
        csv_min.write( '{}\n'.format(e[0]) )
        csv_max.write( '{}\n'.format(e[1]) )

    csv_min.close()
    csv_max.close()

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

    parser.add_argument('--mode', help="p16, p49, r128", type=str, default="r128")
    parser.add_argument("--boundary", type=int, default=0)
    parser.add_argument("--no_blur", action='store_true')
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

    if args.npy and is_video(args.output):
        os.makedirs(os.path.join(os.path.dirname(args.output), BAND + "_npy"), exist_ok=True)

    # load model
    init_model()

    # compute depth maps
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    write_metadata(args.input, data)
