import numpy as np
import torch
import cv2
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

# MiDAS v3.1 
# https://github.com/isl-org/MiDaS
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.model_loader import load_model
from torchvision.transforms import Compose

from common.io import create_folder, to_float_rgb, write_depth, check_overwrite
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import heat_to_rgb

BAND = "depth_midas"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
MODEL = 'models/dpt_beit_large_512.pt'
WIDTH = int(1280)
HEIGHT = int(720)

# Limit for the GPU (NVIDIA RTX 2080), can be adjusted 
GPU_threshold = 1600 - 32 

device = None
model = None
transform = None
data = None


# Load MiDAS v3.1 
def init_model(optimize=True):
    global model, device, transform
    device = torch.device( DEVICE )
    model, transform, net_w, net_h = load_model(device, MODEL, "dpt_beit_large_512", optimize, False, False)

    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    msize = 512
    if msize > GPU_threshold:
        msize = GPU_threshold
    net_w = net_h = msize

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    if optimize==True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)  
        model = model.half()

    model.to(device)
    return model


def infer(img, optimize=True, normalize=True):
    global model, device, transform

    if model == None:
        init_model(optimize)

    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize==True and device==torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=True
                    ).squeeze()
        
    if normalize:
        prediction = prediction.cpu().numpy()

        # Normalization
        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)

    return prediction


def process_image(args):
    # LOAD resource 
    from PIL import Image
    in_image = Image.open(args.input).convert("RGB")
    print("Original size:", in_image.width, in_image.height)

    img = to_float_rgb( in_image )
    result = infer(img, optimize=args.optimize, normalize=False)
    result = result.cpu().numpy()

    if data:
        depth_min = result.min()
        depth_max = result.max()

        data["bands"][BAND]["values"] = { 
                                            "min" : {
                                                    "value": depth_min, 
                                                    "type": "float"
                                            },
                                            "max" : {
                                                "value": depth_max,
                                                "type": "float" }
                                        }

    # Save depth
    write_depth( args.output, result, normalize=True, flip=True, heatmap=True)


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
            
        img = to_float_rgb( in_video[i].asnumpy() )
        prediction = infer(img, optimize=args.optimize,  ormalize=False)
        prediction = prediction.cpu().numpy()

        depth_min = prediction.min()
        depth_max = prediction.max()

        depth = (prediction - depth_min) / (depth_max - depth_min)
        out_video.write( ( heat_to_rgb(depth.astype(np.float64)) * 255 ).astype(np.uint8) )

        if args.subpath != '':
            write_depth( os.path.join(args.subpath, "{:05d}.png".format(i)), prediction, heatmap=True)

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

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")
    parser.add_argument('-model', help="model path", type=str, default=MODEL)

    parser.add_argument('-subpath', '-d', help="subpath to frames", type=str, default='')
    parser.add_argument('-optimize', dest='optimize', action='store_true')
    parser.add_argument('-no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

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
    init_model(args.optimize)

    # compute depth maps
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    if data:
        write_metadata(args.input, data)