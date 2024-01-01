import numpy as np
import argparse
import torch
import json
import os

import warnings
warnings.filterwarnings("ignore")


import cv2

from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.densepose_outputs_vertex import DensePoseOutputsVertexVisualizer, get_texture_atlas, get_texture_atlases
from densepose.vis.extractor import create_extractor, DensePoseResultExtractor, DensePoseOutputsExtractor

# from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances

from common.io import  check_overwrite

BAND = "mask_densepose"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
CONFIG = 'bands/densepose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
MODEL  = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

model = None
device = None
data = None
cfg = None
visualizer = None
extractor = None


def init_model():
    global device, model, cfg, visualizer, extractor, texture_atlas, texture_atlases_dict

    # Load the model
    device = torch.device( DEVICE )

    # Initialize Detectron2 configuration for DensePose
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = MODEL
    cfg.MODEL.DEVICE = DEVICE

    # visualizer = DensePoseOutputsVertexVisualizer(cfg=cfg, alpha=1, )
    visualizer = DensePoseResultsFineSegmentationVisualizer(cfg=cfg, alpha=1)
    extractor = create_extractor(visualizer)

    texture_atlas = get_texture_atlas(None)
    texture_atlases_dict = get_texture_atlases(None)

    # Create a predictor using the trained model    
    model = DefaultPredictor(cfg)


def infer(img):
    global model, cfg, visualizer, extractor

    width = img.shape[1]
    height = img.shape[0]

    with torch.no_grad():
        outputs = model(img)["instances"]

    results = extractor(outputs)

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    out_frame = visualizer.visualize(arr, results)

    return out_frame


def runVideo(args, data = None):
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

    output_path = args.output
    output_filename = os.path.basename(output_path)
    mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

    for f in tqdm( range(total_frames) ):
        img = cv2.cvtColor(in_video[f].asnumpy(), cv2.COLOR_BGR2RGB)
        masks = infer(img)
        mask_video.write( masks.astype(np.uint8) )

    # Mask
    mask_video.close()
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename


def runImage(args, data = None):
    
    # Export properties
    output_path = args.output
    output_filename = os.path.basename(output_path)
    
    img = cv2.imread(args.input)

    # run the inferance
    masks = infer(img)
    cv2.imwrite(args.output, masks.astype(np.uint8))

    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")

    args = parser.parse_args()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join(input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit(".", 1)[0]
    input_extension = input_filename.rsplit(".", 1)[1]
    input_video = input_extension == "mp4"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + "." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, BAND + "." + input_extension)

    check_overwrite(args.output)

    init_model()

    if os.path.isfile(input_payload):
        if not data:
            data = json.load( open(input_payload) )

    # compute depth maps
    if input_video:
        runVideo(args, data)
    else:
        runImage(args, data)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )