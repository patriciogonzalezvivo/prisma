import numpy as np
import torch
import json
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

from common.encode import heat_to_rgb
from common.io import create_folder, write_depth

BAND = "depth_zoe"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
WIDTH = int(1280)
HEIGHT = int(720)

device = None
model = None
data = None

# Load Zoe
def init_model(optimize=True):
    global model, device
    device = torch.device( DEVICE )

    repo = "isl-org/ZoeDepth"
    # Zoe_N
    # conf = get_config("zoedepth", "infer")
    # model = build_model(conf).to(DEVICE)
    model = torch.hub.load(repo, "ZoeD_N", pretrained=True).to(DEVICE)

    # Zoe_K
    # model = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    # model = build_model(conf).to(DEVICE)
    # conf = get_config("zoedepth", "infer", config_version="kitti").to(DEVICE)

    # Zoe_NK
    # conf = get_config("zoedepth_nk", "infer")
    # model = build_model(conf).to(DEVICE)
    # model = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(DEVICE)
    
    return model


def infer(img, msize, optimize=True, normalize=True):
    global model, device

    if model == None:
        init_model(optimize)

    img = img.astype(np.float32)
    prediction = model.infer_pil( img )

    if normalize:
        prediction = prediction.cpu().detach().numpy()

        # Normalization
        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = 1.0-(prediction - depth_min) / (depth_max - depth_min)

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

    width /= 2
    height /= 2

    output_folder = os.path.dirname(args.output)
    output_folder = os.path.join(output_folder, BAND)
    create_folder(output_folder)
    if data:
        data["bands"][BAND]["folder"] = BAND

    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output)

    csv_files = []
    for i in tqdm( range(total_frames) ):
        img = in_video[i].asnumpy() 
        prediction = model.infer_pil( img )
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth = (prediction - depth_min) / (depth_max - depth_min)
        depth = 1.0-depth.astype(np.float64)
        out_video.write( ( heat_to_rgb(depth) * 255 ).astype(np.uint8) )

        write_depth( os.path.join(output_folder, "{:05d}.png".format(i)), prediction, flip=False, heatmap=True)

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

    prediction = model.infer_pil( in_image )

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

    # Save depth
    write_depth( args.output, prediction, flip=False, heatmap=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")
    parser.add_argument('-width', help="final max width", type=str, default=WIDTH)
    parser.add_argument('-height', help="final max height", type=str, default=HEIGHT)

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    init_model()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )
        
    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join( input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    if not input_video:
        input_extension = "png"

    if not data:
        payload_path = os.path.join(input_folder, "payload.json")
        data = json.load( open(payload_path) )

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + "." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, BAND + "." + input_extension)

    # print("output", args.output)
    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    if data:
        data["bands"][BAND] = { "url": output_filename }

    # compute depth maps
    if input_video:
        process_video(args)
    else:
        process_image(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )
