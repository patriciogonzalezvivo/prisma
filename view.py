#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import argparse
import re

from typing import Final

import rerun as rr
import numpy as np
import decord
from tqdm import tqdm
from bands.common.colmap import read_model

DEPTH_IMAGE_SCALING: Final = 1e4

skip = ["rgba_scaled", "perspective_ori", "perspective_lat"]

# calculate focus point from field of view
def calc_focus_point(fov, width):
    fov = fov * np.pi / 180.0
    focal_length = (width * 0.5) / np.tan(fov * 0.5)
    return focal_length

def add_values(data, name, values):
    if "values" in data["bands"][name]:
        for value in data["bands"][name]["values"]:
            if "url" in data["bands"][name]["values"][value]:
                cvs_path = os.path.join(args.input, data["bands"][name]["values"][value]["url"])
                lines = open(cvs_path, "r").readlines()
                type = data["bands"][name]["values"][value]["type"]
                if type == "int":
                    values[value] = [int(line) for line in lines]
                elif type == "float":
                    values[value] = [float(line) for line in lines]
                elif type == "vec2":
                    values[value] = [[float(v) for v in line.split(",")] for line in lines]


def add_band(data, name, values, path="bands/textures/"):

    if name in skip:
        return
            
    width = int(data["width"] / 2)
    height = int(data["height"] / 2)
    frames = int(data["frames"])

    band_video = None
    if "url" in data["bands"][name]:
        if not data["bands"][name]["url"].endswith(".mp4"):
            return
        band_path = os.path.join(args.input, data["bands"][name]["url"])
        band_video = decord.VideoReader(band_path)
                
    frame_idx = 0
    for f in tqdm( range( frames ), desc=name ):
        rr.set_time_sequence("frame", frame_idx)

        if band_video is not None:
            band = band_video[f].asnumpy()

            if name == "depth":
                band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                depth_min = 1.0
                depth_max = 10.0
                if "u_depth_min" in values:
                    depth_min = values["u_depth_min"][frame_idx]
                if "u_depth_max" in values:
                    depth_max = values["u_depth_max"][frame_idx]

                band = depth_min + band * (depth_max - depth_min)
                rr.log_depth_image(path + "depth", band * 100.0, meter=DEPTH_IMAGE_SCALING)
            elif name == "mask":
                band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                rr.log_segmentation_image(path + "mask", band)
            else:
                band = cv2.resize(band, dsize=(width, height), interpolation=cv2.INTER_AREA)
                rr.log_image(path + name, band)

        frame_idx += 1


def init(data):
    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )

    rr.log_view_coordinates("bands", up="-Y", timeless=True)
            
    sparse_path = os.path.join( args.input, "sparse", "0" )
    if os.path.isdir( sparse_path ):
        cameras, images, points3D = read_model(sparse_path, ext=".bin")

    # Iterate through images (video frames) logging data related to each frame.
    images_path = os.path.join(args.input, "images")
    if os.path.isdir( sparse_path ):
        for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
            image_file = os.path.join(images_path, image.name)
            if not os.path.exists(image_file):
                continue

            # COLMAP sets image ids that don't match the original video frame
            idx_match = re.search(r"\d+", image.name)
            assert idx_match is not None
            frame_idx = int(idx_match.group(0))

            quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
            camera = cameras[image.camera_id]

            visible = [id != -1 and points3D.get(id) is not None for id in image.point3D_ids]
            visible_ids = image.point3D_ids[visible]

            visible_xyzs = [points3D[id] for id in visible_ids]
            visible_xys = image.xys[visible]

            rr.set_time_sequence("frame", frame_idx - 1)

            points = [point.xyz for point in visible_xyzs]
            point_colors = [point.rgb for point in visible_xyzs]
            point_errors = [point.error for point in visible_xyzs]

            rr.log_scalar("camera/reproj_err", np.mean(point_errors), color=[240, 45, 58])

            rr.log_points("points", points, colors=point_colors, ext={"error": point_errors})

            # COLMAP's camera transform is "camera from world"
            rr.log_transform3d("camera", rr.TranslationRotationScale3D(image.tvec, rr.Quaternion(xyzw=quat_xyzw)), from_parent=True)
            rr.log_view_coordinates("camera", xyz="RDF")  # X=Right, Y=Down, Z=Forward

            # Log camera intrinsics
            if camera.model == "PINHOLE":
                rr.log_pinhole(
                    "camera/rgba",
                    width=camera.width,
                    height=camera.height,
                    focal_length_px=camera.params[:2],
                    principal_point_px=camera.params[2:],
                )
            elif camera.model == "SIMPLE_PINHOLE":
                rr.log_pinhole(
                    "camera/rgba",
                    width=camera.width,
                    height=camera.height,
                    focal_length_px=camera.params[:2],
                    principal_point_px=camera.params[1:],
                )
            rr.log_image_file("camera/rgba", img_path=image_file)
            rr.log_points("camera/rgba/keypoints", visible_xys, colors=[34, 138, 167])
            
    values = {}

    # extract values from bands
    for band in data["bands"]:
        add_values(data, band, values)

    print("Found uniforms:", values.keys())

    # Attempt to reconstruct camera intrinsics
    width = int(data["width"] / 2)
    height = int(data["height"] / 2)
    
    fps = data["fps"]
    u_cen = float(width / 2)
    v_cen = float(height / 2)
    f_len = float(height * width) ** 0.5
    
    if "perspective" in data["bands"]:
        if "u_fov" in data["bands"]["perspective"]:
            fov = data["bands"]["perspective"]["values"]["u_fov"]["value"]
            f_len = calc_focus_point(fov, width, height)
            print("FOV:", fov)
            print("focal length:", f_len)

    print("Setting camera intrinsics to : ", width, height, f_len, u_cen, v_cen)

    frames = data["frames"]
    for frame_idx in tqdm( range( frames ) ):
        rr.set_time_sequence("frame", frame_idx)

        # log float values
        for value in values:
            if isinstance(values[value][frame_idx], float):
                rr.log_scalar("bands/uniforms/" + value, values[value][frame_idx])

        # log camera
        if "u_principal_point" in values:
            pp = values["u_principal_point"][frame_idx]
            rr.log_view_coordinates("bands/textures", xyz="RDF")
            u_cen = int(width/2 + pp[0] * width/2)
            v_cen = int(height/2 + pp[1] * height/2)

        rr.log_pinhole(
            "bands/textures",
            width=width,
            height=height,
            focal_length_px=[f_len, f_len],
            principal_point_px=[u_cen, v_cen]
        )

    # log bands
    for band in data["bands"]:
        add_band(data, band, values)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="Input folder. Ex: `data/000`", type=str, required=True)
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "view")

    init(args)
    rr.script_teardown(args)

    