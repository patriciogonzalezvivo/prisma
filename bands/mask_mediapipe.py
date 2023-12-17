import argparse
import torch
import json
import os

import warnings
warnings.filterwarnings("ignore")

from typing import Any, Final, Iterator

import cv2
import mediapipe as mp
import numpy as np

import snowy


BAND = "mask_mediapipe"

model = None
data = None

def init_model():
    global model
    mp_pose = mp.solutions.pose
    model = mp_pose.Pose(enable_segmentation=True)


# def read_landmark_positions_2d(
#     results: Any,
#     image_width: int,
#     image_height: int,
# ) -> npt.NDArray[np.float32] | None:
#     if results.pose_landmarks is None:
#         return None
#     else:
#         normalized_landmarks = [results.pose_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
#         return np.array([(image_width * lm.x, image_height * lm.y) for lm in normalized_landmarks])


# def read_landmark_positions_3d(
#     results: Any,
# ) -> npt.NDArray[np.float32] | None:
#     if results.pose_landmarks is None:
#         return None
#     else:
#         landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
#         return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    

def infer(img):
    global model

    results = model.process(img)
    h, w, _ = img.shape
    # landmark_positions_2d = read_landmark_positions_2d(results, w, h)
    # landmark_positions_3d = read_landmark_positions_3d(results)
    mask = results.segmentation_mask

    # mask = snowy.rgb_to_luminance( cv2,mer )
    print(mask.shape)

    # gray = snowy.rgb_to_luminance( snowy.extract_rgb(masks * 255) )
    # grey = np.uint8(np.clip(mask * 255, 0, 255))

    # Convert 2D mask into a grayscale image of width, height and channel
    grey = snowy.rgb_to_luminance( cv2.merge((mask,mask,mask)) )

    sdf = snowy.generate_sdf(grey != 0.0)
    sdf = (sdf + 127.0) / 255.0
    sdf = (sdf - 0.25) * 2.0

    return mask, sdf


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
    masks, sdf = infer(img)
        
    # Mask
    cv2.imwrite(args.output, (masks * 255).astype(np.uint8))
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename

    # SDF
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
    mask_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

    sdf_filename = output_basename + "_sdf." + output_extension
    sdf_path = output_folder + "/" + sdf_filename
    sdf_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=sdf_path )

    for f in tqdm( range(total_frames) ):
        img = cv2.cvtColor(in_video[f].asnumpy(), cv2.COLOR_BGR2RGB)

        masks = infer(img)

        mask_video.write( (masks * 255).astype(np.uint8) )

        sdf = np.uint8(np.clip(sdf * 255, 0, 255))
        sdf = cv2.merge((sdf,sdf,sdf))
        sdf_video.write( sdf )

    # Mask
    mask_video.close()
    data["bands"][BAND] = { }
    data["bands"][BAND]["url"] = output_filename

    # SDF
    sdf_video.close()
    data["bands"][BAND + "_sdf"] = { }
    data["bands"][BAND + "_sdf"]["url"] = sdf_filename


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