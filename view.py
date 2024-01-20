# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#

import os
import cv2
import argparse
import re

from typing import Final

import rerun as rr

import numpy as np
import numpy.typing as npt
from typing import Final

import decord
from tqdm import tqdm
from bands.common.colmap import Camera, read_model
from bands.common.meta import load_metadata
from bands.common.encode import rgb_to_heat

ROOT = "bands/"
DEPTH_IMAGE_SCALING: Final = 1e4
DEPTH_SCALING = {
    "depth_midas": 7.0,
    "depth_marigold": 0.3,
    "depth_zoedepth": 1.0,
    "depth_patchfusion": 1.0,
}
FILTER_MIN_VISIBLE: Final = 500


def scale_camera(camera: Camera, resize: tuple[int, int]) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    if camera.model == "PINHOLE":
        new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:] * scale_factor)
    elif camera.model == "SIMPLE_PINHOLE":
        new_params = np.append(camera.params[:1] * scale_factor, camera.params[2:] * scale_factor)

    return (Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)


# From rerun's example 
# https://github.com/rerun-io/rerun/blob/bef0d7851e6e3d49aa0ccd40ba59846b37c97a81/examples/python/structure_from_motion/main.py
def read_and_log_sparse_reconstruction(args, filter_output: bool, resize = None) -> bool:
    # COLMAP's sparse reconstruction
    sparse_path = os.path.join( args.input, "sparse", "0" )

    if os.path.isdir( sparse_path ) and os.path.isdir( sparse_path ):
        cameras, images, points3D = read_model(sparse_path, ext=".bin")

        if filter_output:
            # Filter out noisy points
            points3D = {id: point for id, point in points3D.items() if point.rgb.any() and len(point.image_ids) > 4}

        for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
            
            # COLMAP sets image ids that don't match the original video frame
            idx_match = re.search(r"\d+", image.name)
            assert idx_match is not None
            frame_idx = int(idx_match.group(0))

            quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
            camera = cameras[image.camera_id]
            if resize:
                camera, scale_factor = scale_camera(camera, resize)
            else:
                scale_factor = np.array([1.0, 1.0])

            visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]
            visible_ids = image.point3D_ids[visible]

            if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
                continue

            visible_xyzs = [points3D[id] for id in visible_ids]
            visible_xys = image.xys[visible]
            if resize:
                visible_xys *= scale_factor

            rr.set_time_sequence("frame", frame_idx)

            points = [point.xyz for point in visible_xyzs]
            point_colors = [point.rgb for point in visible_xyzs]
            point_errors = [point.error for point in visible_xyzs]

            rr.log(ROOT + "avg_reproj_err", rr.TimeSeriesScalar(np.mean(point_errors), color=[240, 45, 58]))

            rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))

            # COLMAP's camera transform is "camera from world"
            rr.log(ROOT, rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True) )
            rr.log(ROOT, rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward

            # Log camera intrinsics
            rr.log(
                ROOT,
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=camera.params[:2],
                    principal_point=camera.params[2:],)
            )

            rr.log(ROOT + "keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))

            # Iterate through images (video frames) logging data related to each frame.
            # images_path = os.path.join(args.input, "images")

            # image_file = os.path.join(images_path, image.name)
            # if not os.path.exists(image_file):
            #     continue

            # if resize:
            #     bgr = cv2.imread(str(image_file))
            #     bgr = cv2.resize(bgr, resize)
            #     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            #     rr.log(ROOT + "rgba", rr.Image(rgb).compress(jpeg_quality=75))
            # else:
            #     rr.log(ROOT + "rgba", rr.ImageEncoded(path=image_file))


        return True
    return False


def extract_values(data):
    values = {}

    for band in data["bands"]:

        if "values" not in data["bands"][band]:
            continue
        
        for value in data["bands"][band]["values"]:
            address = band + "_" + value

            # load values from file (video)
            if "url" in data["bands"][band]["values"][value]:
                cvs_path = os.path.join(args.input, data["bands"][band]["values"][value]["url"])
                lines = open(cvs_path, "r").readlines()
                t = data["bands"][band]["values"][value]["type"]

                if t == "int":
                    values[address] = [int(line) for line in lines]
                elif t == "float":
                    values[address] = [float(line) for line in lines]
                elif t == "vec2":
                    values[address] = [[float(v) for v in line.split(",")] for line in lines]

                for i in range(len(values[address])):
                    rr.set_time_sequence("frame", i)
                    rr.log(ROOT + address, rr.TimeSeriesScalar(values[address][i]))
                    
            # load value from json (image)
            elif "value" in data["bands"][band]["values"][value]:

                rr.set_time_sequence("frame", 0)

                t = data["bands"][band]["values"][value]["type"]
                value = data["bands"][band]["values"][value]["value"]

                if t == "int":
                    value = int(value)
                
                elif t == "float":
                    value = float(value)

                elif t == "vec2":
                    value = [float(v) for v in value.split(",")]

                if value is not None:
                    rr.log(ROOT + address, rr.TimeSeriesScalar(value))
                    values[address] = value

    return values


def add_band_image(data, band, img, resize=None, index=None):
    if resize is None:
        resize = (int(data["width"]), int(data["height"]))
    
    img = cv2.resize(img, resize)

    if band.startswith("depth"):
        depth_img = rgb_to_heat(img)
        depth_min = 1.0
        depth_max = 10.0

        if index is not None:
            if band + "_min" in data["values"]:
                depth_min = data["values"][band + "_min"][index]
            if band + "_max" in data["values"]:
                depth_max = data["values"][band + "_max"][index]

        else:
            if band + "_min" in data["values"]:
                depth_min = data["values"][band + "_min"]
            if band + "_max" in data["values"]:
                depth_max = data["values"][band + "_max"]

        depth_img = depth_min + depth_img * (depth_max - depth_min)
        rr.log(ROOT + band, rr.DepthImage(depth_img, meter=DEPTH_SCALING[band]))

    else:
        rr.log(ROOT + band, rr.Image(img).compress(jpeg_quality=95))


def add_band(data, band, resize):
    if "url" in data["bands"][band]:

        # If it's a video
        if data["bands"][band]["url"].endswith(".mp4"):
            print("About to load band video", band, "at", data["bands"][band]["url"])
            frames = int(data["frames"])

            band_path = os.path.join(args.input, data["bands"][band]["url"])
            band_video = decord.VideoReader(band_path)

            frame_idx = 0
            for f in tqdm( range( frames ), desc=band ):
                rr.set_time_sequence("frame", frame_idx)

                band_frame = band_video[f].asnumpy()
                add_band_image(data, band, band_frame, index=frame_idx, resize=resize)
                frame_idx += 1

        # If it's an image
        elif data["bands"][band]["url"].endswith(".png") or data["bands"][band]["url"].endswith(".jpg"):
            print("About to load band image", band, "at", data["bands"][band]["url"])
            rr.set_time_sequence("frame", 0)
            band_path = os.path.join(args.input, data["bands"][band]["url"])
            band_img = cv2.cvtColor(cv2.imread(band_path), cv2.COLOR_BGR2RGB)
            add_band_image(data, band, band_img, resize=resize)


def init(args):
    data = load_metadata(args.input)

    # extract values from bands
    data["values"] = extract_values(data)

    width = int(data["width"])
    height = int(data["height"])
    resize = (int(width * args.scale), int(height * args.scale))
    u_cen = data["principal_point"][0] * args.scale
    v_cen = data["principal_point"][1] * args.scale
    f_len = data["focal_length"] * args.scale
    f_v = data["field_of_view"] * args.scale

    rr.log("bands", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    
    sparse = False
    frames = 1
    if "frames" in data:
        frames = data["frames"]
        
        # Extract values from COLMAP's sparse reconstruction
        sparse = read_and_log_sparse_reconstruction(args, filter_output=True, resize=resize)

    if not sparse:
        for frame_idx in tqdm( range( frames ) ):
            rr.set_time_sequence("frame", frame_idx)

            rr.log(
                ROOT,
                rr.Pinhole(
                        resolution=[resize[0], resize[1]],
                        focal_length=[f_len, f_len],
                        principal_point=[u_cen, v_cen],
                    )
            )

    # log bands
    for band in data["bands"]:
        add_band(data, band, resize=resize)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input folder. Ex: `data/000`", type=str, required=True)
    parser.add_argument('--scale', '-s', help="Scale factor. Ex: `0.5`", type=float, default=0.5)
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "view")

    init(args)
    rr.script_teardown(args)

    