import numpy as np
import torch
import json
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

from common.encode import heat_to_rgb
from common.io import create_folder, write_depth

BAND = "depth_patchfusion"
MODEL = "models/patchfusion_u4k.pt"
# MODEL_TYPE = "zoedepth"
MODEL_TYPE = "zoedepth_custom"
CONFIG = "bands/patchfusion/zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
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
    overwrite_kwargs["model"] = "zoedepth_custom"

    config = get_config_user("zoedepth_custom", **overwrite_kwargs)
    config["pretrained_resource"] = ''
    model = build_model(config)
    model = load_ckpt(model, MODEL)
    model.eval()
    model.cuda()
    
    return model


def infer(img, img_resolution, mode="r128", blr_mask=True, boundary=0, normalize=True):
    global model, device, transform

    if model == None:
        init_model()

    crop_size = (int(img_resolution[0] // 4), int(img_resolution[1] // 4))

    # Load image from dataset
    img = torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2) # shape: 1, 3, h, w
    img_lr = transform(img)

    avg_depth_map = regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=0, offset_y=0, img_lr=img_lr)

    if mode== 'p16':
        pass
    elif mode== 'p49':
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=crop_size[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=0, offset_y=crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=crop_size[1]//2, offset_y=crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

    elif mode[0] == 'r':
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=crop_size[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=0, offset_y=crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)
        regular_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, offset_x=crop_size[1]//2, offset_y=crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

        for i in tqdm(range(int(mode[1:]))):
            random_tile(model, img, img_resolution, crop_size=crop_size, transform=transform, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=blr_mask)

    return avg_depth_map.average_map


# def process_video(args):
#     import decord
#     from common.io import VideoWriter

#     # LOAD resource 
#     in_video = decord.VideoReader(args.input)
#     width = in_video[0].shape[1]
#     height = in_video[0].shape[0]
#     total_frames = len(in_video)
#     fps = in_video.get_avg_fps()

#     width /= 2
#     height /= 2

#     output_folder = os.path.dirname(args.output)
#     output_folder = os.path.join(output_folder, BAND)
#     create_folder(output_folder)
#     if data:
#         data["band"][BAND]["folder"] = BAND

#     out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output)

#     csv_files = []
#     for i in tqdm( range(total_frames) ):
#         img = in_video[i].asnumpy() 
#         prediction = model.infer_pil( img )
#         depth_min = prediction.min()
#         depth_max = prediction.max()
#         depth = (prediction - depth_min) / (depth_max - depth_min)
#         depth = 1.0-depth.astype(np.float64)
#         out_video.write( ( heat_to_rgb(depth) * 255 ).astype(np.uint8) )

#         write_depth( os.path.join(output_folder, "{:05d}.png".format(i)), prediction, flip=False, heatmap=True)

#         csv_files.append( ( depth_min.item(),
#                             depth_max.item()  ) )
#     out_video.close()

#     output_folder = os.path.dirname(args.output)
#     csv_min = open( os.path.join( output_folder, "depth_min.csv" ) , 'w')
#     csv_max = open( os.path.join( output_folder, "depth_max.csv" ) , 'w')

#     for e in csv_files:
#         csv_min.write( '{}\n'.format(e[0]) )
#         csv_max.write( '{}\n'.format(e[1]) )

#     csv_min.close()
#     csv_max.close()

#     if data:
#         data["band"][BAND]["values"] = { 
#                                             "min" : {
#                                                     "type": "float",
#                                                     "url": "depth_min.csv"
#                                             },
#                                             "max" : {
#                                                 "type": "float", 
#                                                 "url": "depth_max.csv",
#                                             }
#                                         }


def process_image(args):
    # # LOAD resource 
    # from PIL import Image
    # in_image = Image.open(args.input).convert("RGB")

    in_image = cv2.imread(args.input)
    if in_image.ndim == 2:
        in_image = cv2.cvtColor(in_image, cv2.COLOR_GRAY2BGR)
    in_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB) / 255.0
    print(in_image.shape)
    in_image_resolution = (in_image.shape[0], in_image.shape[1])

    in_image = F.interpolate(torch.tensor(in_image).unsqueeze(dim=0).permute(0, 3, 1, 2), in_image_resolution, mode='bicubic', align_corners=True)
    in_image = in_image.squeeze().permute(1, 2, 0)

    prediction = infer( in_image, in_image_resolution )

    if data:
        depth_min = prediction.min().item()
        depth_max = prediction.max().item()
        data["band"][BAND]["values"] = { 
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

    init_model()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["band"]["rgba"]["url"] )
            global IMG_RESOLUTION
            IMG_RESOLUTION = (data["height"], data["width"])
        
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
        data["band"][BAND] = { "url": output_filename }

    # compute depth maps
    if input_video:
        process_video(args)
    else:
        process_image(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )
