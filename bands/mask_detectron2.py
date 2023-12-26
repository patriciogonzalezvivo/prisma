import numpy as np
import argparse
import torch
import json
import os

import warnings
warnings.filterwarnings("ignore")

import cv2

import torch
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import snowy

from common.encode import hue_to_rgb, mask_to_rgb
from common.io import VideoWriter

BAND = "mask"
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
CONFIG = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
COCO = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

model = None
device = None
data = None

def init_model():
    global device, model

    # Load the model
    device = torch.device( DEVICE )

    # Create a predictor using the ZOO model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG)

    # Create a predictor using the trained model    
    model = DefaultPredictor(cfg)


def infer(img):
    global model

    # run inferance
    result = model(img)
    result = result["instances"].to("cpu");
    scores = result.scores if result.has("scores") else None
    classes = result.pred_classes.tolist() if result.has("pred_classes") else None
    class_list = model.metadata.get("thing_classes", None)  

    # prepare
    masks = np.zeros(img.shape)
    seg = np.zeros(img.shape)
    seg_ids = {}
    total_classes = len(class_list)

    if result.has("pred_masks"):
        masks_bin = np.asarray(result.pred_masks)
        total_masks = len(masks_bin)
        for i in range( total_masks ):
            mask = mask_to_rgb(masks_bin[i])
            mask_class = classes[i]
            mask_class_name = class_list[mask_class]
            mask_score = scores[i]

            # print(i, mask_class, mask_class_name, mask_score)

            # class_index = class_list.index(mask_class)
            hue = float(mask_class) / float(total_classes)
            color = hue_to_rgb( hue )
            seg_ids[mask_class_name] = { 'hue': hue , "rgb": color.tolist() }

            if mask_class_name in COCO:
                masks = masks + mask
            
            seg = seg + color * mask

    mask = snowy.rgb_to_luminance( snowy.extract_rgb(masks) )
    sdf = snowy.generate_sdf(mask != 0.0)
    sdf = (sdf + 127.0) / 255.0
    sdf = (sdf - 0.25) * 2.0

    return masks, seg, sdf


def runImage(args, data = None):
    
    # Export properties
    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    seg_filename = output_basename + "_seg." + output_extension
    seg_path = output_folder + "/" + seg_filename

    sdf_filename = output_basename + "_sdf." + output_extension
    sdf_path = output_folder + "/" + sdf_filename
    
    img = cv2.imread(args.input)

    # run the inferance
    masks, seg, sdf = infer(img)
        
    # Mask
    if args.mask:
        cv2.imwrite(args.output, masks.astype(np.uint8))
        data["bands"][BAND] = { }
        data["bands"][BAND]["url"] = output_filename

    # Seg
    if args.seg:
        cv2.imwrite(seg_path, seg.astype(np.uint8))
        data["bands"][BAND + "_seg"] = { }
        data["bands"][BAND + "_seg"]["url"] = sdf_filename
        data["bands"][BAND + "_seg"]["ids"] = COCO

    # SDF
    if args.sdf:
        snowy.export(sdf, sdf_path)
        data["bands"][BAND + "_sdf"] = { }
        data["bands"][BAND + "_sdf"]["url"] = sdf_filename


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
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    if args.mask:
        mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

    if args.sdf:
        sdf_filename = output_basename + "_sdf." + output_extension
        sdf_path = output_folder + "/" + sdf_filename
        sdf_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=sdf_path )

    if args.seg:
        seg_filename = output_basename + "_seg." + output_extension
        seg_path = output_folder + "/" + seg_filename
        seg_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=seg_path )

    for f in tqdm( range(total_frames) ):
        img = cv2.cvtColor(in_video[f].asnumpy(), cv2.COLOR_BGR2RGB)

        masks, seg, sdf = infer(img)

        if args.mask:
            mask_video.write( masks.astype(np.uint8) )

        if args.seg:
            seg_video.write( seg.astype(np.uint8) )

        if args.sdf:
            sdf = np.uint8(np.clip(sdf * 255, 0, 255))
            sdf = cv2.merge((sdf,sdf,sdf))
            sdf_video.write( sdf )

    # Mask
    if args.mask:
        mask_video.close()
        data["bands"][BAND] = { }
        data["bands"][BAND]["url"] = output_filename

    # SEG
    if args.seg:
        seg_video.close()
        data["bands"][BAND + "_seg"] = { }
        data["bands"][BAND + "_seg"]["url"] = seg_filename
        data["bands"][BAND + "_seg"]["ids"] = COCO

    # SDF
    if args.sdf:
        sdf_video.close()
        data["bands"][BAND + "_sdf"] = { }
        data["bands"][BAND + "_sdf"]["url"] = sdf_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")

    parser.add_argument("-sdf", action='store_true')
    parser.add_argument("-seg", action='store_true')
    parser.add_argument("-mask", action='store_true')


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