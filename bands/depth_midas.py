import numpy as np
import torch
import cv2
import json
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

# MiDAS v3.1 
# https://github.com/isl-org/MiDaS
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.model_loader import load_model
from torchvision.transforms import Compose

from common.encode import heat_to_rgb
from common.io import create_folder, to_float_rgb, write_depth, check_overwrite

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

    if args.subpath != '':
        if data:
            data["bands"][BAND]["folder"] = args.subpath
        args.subpath = os.path.join(output_folder, args.subpath)
        create_folder(args.subpath)

    csv_files = []
    for i in tqdm( range(total_frames) ):
            
        img = to_float_rgb( in_video[i].asnumpy() )
        prediction = infer(img, args.optimize, False)
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

    output_folder = os.path.dirname(args.output)
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

def process_image(args):
    
    # LOAD resource 
    from PIL import Image
    in_image = Image.open(args.input).convert("RGB")
    print("Original size:", in_image.width, in_image.height)

    img = to_float_rgb( in_image )
    result = infer(img, args.optimize, False)

    if data:
        depth_min = result.min().item()
        depth_max = result.max().item()

        data["bands"][BAND]["values"] = { 
                                                "min" : {
                                                        "value": depth_min, 
                                                        "type": "float"
                                                },
                                                "max" : {
                                                    "value": depth_max,
                                                    "type": "float" }
                                            }

    result = result.cpu().numpy()
        
    # Save depth
    write_depth( args.output, result, heatmap=True)


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

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )
        
    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join(input_folder, "payload.json")
    if os.path.isfile(input_payload):
        data = json.load( open(input_payload) )
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    if not input_video:
        input_extension = "png"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + "." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, BAND + "." + input_extension)

    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    check_overwrite(output_path)

    if data:
        data["bands"][BAND] = { "url": output_filename }

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Load MiDAS v3.1
    init_model(args.optimize)

    # compute depth maps
    if input_video:
        process_video(args)
    else:
        process_image(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )