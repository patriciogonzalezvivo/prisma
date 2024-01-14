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

import os
import cv2
import argparse
import re

from typing import Final

import rerun as rr
import numpy as np
import decord
from tqdm import tqdm
from bands.common.colmap import read_model
from bands.common.meta import load_metadata
from bands.common.encode import rgb_to_heat

DEPTH_IMAGE_SCALING: Final = 1e4

# calculate focus point from field of view
def calc_focus_point(fov, width):
    fov = fov * np.pi / 180.0
    focal_length = (width * 0.5) / np.tan(fov * 0.5)
    return focal_length

def extract_values(data):
    values = {}

    for band in data["bands"]:

        if "values" not in data["bands"][band]:
            continue
        
        for value in data["bands"][band]["values"]:
            address = band + "_" + value

            print("Extracting value:", address)

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
                    rr.log("bands/values/" + address, rr.TimeSeriesScalar(values[address][i]), time=i)
                    
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
                    rr.log("bands/values/" + address, rr.TimeSeriesScalar(value))
                    values[address] = value

    return values


def add_band_image(data, band, img, path="bands/textures/"):
    if band.startswith("depth"):
        depth_img = rgb_to_heat(img)
        depth_min = 1.0
        depth_max = 10.0

        if band + "_min" in data["values"]:
            depth_min = data["values"][band + "_min"]
        if band + "_max" in data["values"]:
            depth_max = data["values"][band + "_max"]

        depth_img = depth_min + depth_img * (depth_max - depth_min)
        rr.log(path + band, rr.DepthImage(depth_img, meter=DEPTH_IMAGE_SCALING))

    else:
        rr.log(path + band, rr.Image(img).compress(jpeg_quality=95))


def add_band(data, band, path="bands/textures/"):

    if "url" in data["bands"][band]:

        # If it's a video
        if data["bands"][band]["url"].endswith(".mp4"):
            frames = int(data["frames"])

            band_path = os.path.join(args.input, data["bands"][band]["url"])
            band_video = decord.VideoReader(band_path)
                
            frame_idx = 0
            for f in tqdm( range( frames ), desc=band ):
                rr.set_time_sequence("frame", frame_idx)
                band_frame = band_video[f].asnumpy()
                add_band_image(data, band, band_frame, path=path)
                frame_idx += 1

        # If it's an image
        else:
            rr.set_time_sequence("frame", 0)

            band_path = os.path.join(args.input, data["bands"][band]["url"])
            band_img = cv2.cvtColor(cv2.imread(band_path), cv2.COLOR_BGR2RGB)
            add_band_image(data, band, band_img, path=path)


def init(args):
    data = load_metadata(args.input)

    rr.log("bands", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    # Iterate through images (video frames) logging data related to each frame.
    images_path = os.path.join(args.input, "images")
    
    # COLMAP's sparse reconstruction
    sparse_path = os.path.join( args.input, "sparse", "0" )
    if os.path.isdir( sparse_path ) and os.path.isdir( sparse_path ):
        cameras, images, points3D = read_model(sparse_path, ext=".bin")
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

            # rr.log_scalar("camera/reproj_err", np.mean(point_errors), color=[240, 45, 58])

            rr.log("points", rr.Points3D(points, colors=point_colors, ext={"error": point_errors}))

            # COLMAP's camera transform is "camera from world"
            rr.log_transform3d("camera", rr.TranslationRotationScale3D(image.tvec, rr.Quaternion(xyzw=quat_xyzw)), from_parent=True)
            rr.log_view_coordinates("camera", xyz="RDF")  # X=Right, Y=Down, Z=Forward

            # Log camera intrinsics
            if camera.model == "PINHOLE":
                rr.log(
                    "camera/rgba",
                    rr.Pinhole(
                        resolution=[camera.width, camera.height],
                        focal_length=camera.params[:2],
                        principal_point=camera.params[2:],)
                )
            elif camera.model == "SIMPLE_PINHOLE":
                rr.log(
                    "camera/rgba",
                    rr.Pinhole(
                        resolution=[camera.width, camera.height],
                        focal_length_px=camera.params[:2],
                        principal_point_px=camera.params[1:],
                    )
                )
            rr.log("camera/rgba", rr.ImageEncoded(path=image_file))
            rr.log("camera/rgba/keypoints", rr.Points3D(visible_xys, colors=[34, 138, 167]))
            
    # extract values from bands
    data["values"] = extract_values(data)

    # Attempt to reconstruct camera intrinsics
    width = int(data["width"])
    height = int(data["height"])
    
    u_cen = float(width / 2)
    v_cen = float(height / 2)
    f_len = float(height * width) ** 0.5
    
    # if "perspective" in data["bands"]:
    #     if "u_fov" in data["bands"]["perspective"]:
    #         fov = data["bands"]["perspective"]["values"]["u_fov"]["value"]
    #         f_len = calc_focus_point(fov, width, height)
    #         print("FOV:", fov)
    #         print("focal length:", f_len)

    # print("Setting camera intrinsics to : ", width, height, f_len, u_cen, v_cen)

    if "frames" in data:
        # fps = data["fps"]
        frames = data["frames"]
        for frame_idx in tqdm( range( frames ) ):
            rr.set_time_sequence("frame", frame_idx)

            # log float values
            for value in data["values"]:
                if isinstance(data["values"][value][frame_idx], float):
                    rr.log("values" + value, data["values"][value][frame_idx])

            # log camera
            if "principal_point" in data["values"]:
                pp = data["values"]["principal_point"][frame_idx]
                rr.log_view_coordinates("bands/textures", xyz="RDF")
                u_cen = int(width/2 + pp[0] * width/2)
                v_cen = int(height/2 + pp[1] * height/2)

            rr.log(
                "bands/textures",
                rr.Pinhole(
                        resolution=[width, height],
                        focal_length=[f_len, f_len],
                        principal_point=[u_cen, v_cen],
                    )
            )

    # log bands
    for band in data["bands"]:
        add_band(data, band)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="Input folder. Ex: `data/000`", type=str, required=True)
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "view")

    init(args)
    rr.script_teardown(args)

    