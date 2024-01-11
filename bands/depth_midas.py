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

import warnings
warnings.filterwarnings("ignore")

from common.io import open_image, create_folder, check_overwrite, write_depth, write_pcl
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import heat_to_rgb

BAND = "depth_midas"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

device = None
model = None
transform = None
data = None
MODELS_VERSIONS = ["midas2-small", "midas2", "midas3-small", "midas3"]


# Load MiDAS v3.1 
def init_model(model_version="midas3"):
    global model, device, transform
    device = torch.device( DEVICE )

    if model_version == "midas2" or model_version == "midas2-small":
        model = torch.hub.load("intel-isl/MiDaS", 'MiDaS')
    else:
        model = torch.hub.load("intel-isl/MiDaS", 'DPT_Large')

    if model_version == "midas2-small" or model_version == "midas3-small":
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    else:
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

    model.to(device)
    model.eval()
    return model


def infer(img, model_version, normalize=False):
    global model, device, transform

    if model == None:
        init_model(model_version)

    img = np.array(img)
    img_input = transform(img).to(device)
    with torch.no_grad():
        prediction = model.forward(img_input)
        prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=True
                    ).squeeze()
    prediction = prediction.cpu().numpy().astype(np.float32)
        
    if normalize:
        # Normalization
        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)

    return prediction


def process_image(args):
    # LOAD resource 
    in_image = open_image(args.input)
    output_folder = os.path.dirname(args.output)

    prediction = infer(in_image, model_version=args.model, normalize=False)
    prediction = prediction.astype(np.float32)

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
                                                "type": "float" }
                                        }

    if args.npy:
        np.save( os.path.join(output_folder, BAND + '.npy'), prediction)

    if args.ply:
        write_pcl( os.path.join(output_folder, BAND + '.ply'), prediction, np.array(in_image), flip=True)

    # Save depth
    write_depth( args.output, prediction, normalize=True, flip=True, heatmap=True, encode_range=True)


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

    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )
    output_folder = os.path.dirname(args.output)

    if args.subpath != '':
        if data:
            data["bands"][BAND]["folder"] = args.subpath
        args.subpath = os.path.join(output_folder, args.subpath)
        create_folder(args.subpath)

    csv_files = []
    for i in tqdm( range(total_frames) ):

        img = in_video[i].asnumpy()
        prediction = infer(img, model_version=args.model, normalize=False)

        if args.npy:
            if args.subpath != '':
                np.save( os.path.join(args.subpath, "{:05d}.npy".format(i)), prediction)
            else:
                np.save( os.path.join(os.path.join(output_folder, BAND + '_npy', prediction), '%04d.npy' % i), prediction)

        depth_min = prediction.min()
        depth_max = prediction.max()

        depth = (prediction - depth_min) / (depth_max - depth_min)
        depth = 1.0-depth
        out_video.write( ( heat_to_rgb(depth) * 255 ).astype(np.uint8) )

        if args.subpath != '':
            write_depth( os.path.join(args.subpath, "{:05d}.png".format(i)), prediction, normalize=True, flip=True, heatmap=True, encode_range=True)

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

    parser.add_argument('--model', type=str, choices=MODELS_VERSIONS, default="midas3")

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

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Load MiDAS v3.1
    init_model(args.model)

    # compute depth maps
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    write_metadata(args.input, data)