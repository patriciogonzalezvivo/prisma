# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#

import os
import sys
import subprocess

import shutil
import argparse
import numpy as np

from common.meta import load_metadata, is_video, get_url, get_target, write_metadata
from common.io import check_overwrite, create_folder, get_check_overwrite

BAND = "camera_pose"
data = None

def process_video(args):

    if get_check_overwrite(args.sparse_folder):
        create_folder(args.sparse_folder)

        # Extract features
        feature_extractor_command = [
            "colmap", "feature_extractor",
            "--database_path", args.database_path,
            "--image_path", args.rgba_folder,
            "--ImageReader.camera_model", args.colmap_camera_model,
            "--ImageReader.single_camera", str(1),
            # "--ImageReader.camera_params", args.colmap_camera_params,
            "--SiftExtraction.use_gpu", str(1),
            # "--SiftExtraction.estimate_affine_shape=true",
            # "--SiftExtraction.domain_size_pooling=true",
            "--SiftExtraction.first_octave", str(0),
        ]

        if args.mask_folder:
            feature_extractor_command += [
                "--ImageReader.mask_path", args.mask_folder,
            ]
        subprocess.run(feature_extractor_command)

        # Match features
        matcher_command = [
            "colmap", 
            args.colmap_matcher + "_matcher",
            "--database_path", args.database_path,
            "--SiftMatching.use_gpu", str(1),
            "--SiftMatching.guided_matching=true",
        ]
        subprocess.run(matcher_command)

        # Reconstruct sparse
        if os.path.exists(args.sparse_folder):
            mapper_command = [
                "colmap", "mapper",
                "--database_path", args.database_path,
                "--image_path", args.rgba_folder,
                "--output_path", args.sparse_folder,
                "--Mapper.multiple_models", str(0), 
                "--Mapper.num_threads", str(16),
                "--Mapper.init_min_tri_angle", str(4),
                "--Mapper.extract_colors", str(1),
                "--Mapper.ba_local_max_refinements", str(1),
                "--Mapper.ba_global_max_refinements", str(1),
                "--Mapper.ba_global_function_tolerance=0.000001",
            ]
            subprocess.run(mapper_command)
            # try:
            #     subprocess.run(mapper_command, check=True)
            # except subprocess.CalledProcessError:
            #     print("mapper failed. relaxing search for initial pair.")
            #     mapper_command += [
            #         "--Mapper.init_max_forward_motion", str(1.0),
            #         "--Mapper.init_min_triangle", str(4),
            #     ]
            #     subprocess.run(mapper_command, check=True)

    if os.path.exists(args.sparse_folder):
        if args.colmap_refine:
            
            # Bundle adjustment
            # if args.colmap_refine:
            bundel_adjuster_command = [
                "colmap",
                "bundle_adjuster",
                "--input_path", args.sparse_folder + "/0",
                "--output_path", args.sparse_folder + "/0",
                "--BundleAdjustment.refine_principal_point", str(1)
            ]
            subprocess.run(bundel_adjuster_command)

        # Undistort images (needed for Gaussian Splatting)
        if args.colmap_undistort:
            image_undistorter_command = [
                "colmap", "image_undistorter",
                "--image_path", args.rgba_folder,
                "--input_path", args.sparse_folder + "/0",
                "--output_path", args.undistorted_folder,
                "--output_type", "COLMAP",
                # "--max_image_size", str(1500),
            ]
            subprocess.run(image_undistorter_command)

            files = os.listdir(args.undistorted_folder)
            for file in files:
                if file == '0':
                    continue
                source_file = os.path.join(args.undistorted_folder, file)
                destination_file = os.path.join(args.sparse_folder + "/0", file)
                shutil.move(source_file, destination_file)

            # Convert to txt
            model_converter_command = [
                "colmap", "model_converter",
                "--input_path", args.sparse_folder + "/0",
                "--output_path", args.sparse_folder + "/0",
                "--output_type", "TXT"
            ]
            subprocess.run(model_converter_command)

    # if there are no images on sparse folder, copy rgba images
    # if len(os.listdir(args.sparse_folder)) == 1:
    #     # copy rgba images to sparse folder
    #     for f in os.listdir(args.rgba_folder):
    #         shutil.copy(os.path.join(args.rgba_folder, f), os.path.join(args.sparse_folder, f))


def convert_to_csv(args):
    sparsedir = args.sparse_folder
    expected_N = len(os.listdir(args.rgba_folder))

    if os.path.exists(os.path.join(sparsedir, "0")):
        sparsedir = os.path.join(sparsedir, "0")

    sys.path.append(".")
    from bands.common.colmap import read_model
    import numpy as np
    cameras, model_images, points3D = read_model(path=sparsedir)

    keys = list(model_images.keys())
    keys = sorted(keys, key=lambda x: model_images[x].name)

    print(len(keys))
    if expected_N is not None:
        assert len(keys) == expected_N

    camkey = model_images[keys[0]].camera_id
    # for key in keys:
    #     print(model_images[key].camera_id)
        # assume single camera setup since we are dealing with videos
        # assert model_images[key].camera_id == camkey

    cam = cameras[camkey]
    params = cam.params

    data["model"] = cam.model
    width = params[0]
    height = params[1]

    if cam.model == "SIMPLE_PINHOLE":
        data["focal_length"] = params[0]
        data["principal_point"] = params[:2].tolist()
    elif cam.model == "PINHOLE":
        data["focal_length"] = params[0]
        data["principal_point"] = params[:2].tolist()
    data["field_of_view"] = 2 * np.arctan(0.5 * params[1] / data["focal_length"]) * 180 / np.pi

    # assert cam.model in ["RADIAL", "SIMPLE_RADIAL"]
    K = np.array([[params[0], 0.0, params[1]],
                  [0.0, params[0], params[2]],
                  [0.0,       0.0,       1.0]])

    Rs = np.stack([model_images[k].qvec2rotmat() for k in keys])
    ts = np.stack([model_images[k].tvec for k in keys])

    N = Rs.shape[0]
    params = params[:3][None].repeat(N, axis=0)
    Rs = Rs.reshape(N, 9)

    lines = np.concatenate((params, Rs, ts), axis=1)

    np.savetxt(args.output, lines, delimiter=",", newline="\n",
               header=(",".join(["f", "ox", "oy"]+
                                [f"R[{i//3},{i%3}]" for i in range(9)]+
                                [f"t[{i}]" for i in range(3)])))
                                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='input video path', required=True)
    parser.add_argument('--output', '-o', type=str, help='where to store data', default='')
    parser.add_argument('--force', '-f', action="store_true", help="Force re-computation of all results.")
    parser.add_argument('--colmap_db', default="colmap.db", help="colmap database filename")
    parser.add_argument('--colmap_camera_model', default="SIMPLE_PINHOLE", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
    parser.add_argument('--colmap_camera_params', default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
    parser.add_argument('--colmap_matcher', default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
    parser.add_argument('--colmap_refine','-r',  help="adjust bundle by refining cameras", action='store_true')
    parser.add_argument('--colmap_undistort','-u',  help="undistort images", action='store_true')
    parser.add_argument('--subpath', '-d', default='sparse', help="Use subpath")
    args = parser.parse_args()

    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.input = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output, force_extension="csv")

    if not is_video(args.input):
        print("Input is not a video. Can't perform colmap reconstruction.")

    input_folder = os.path.dirname(args.input)
    args.mask_folder = os.path.join(input_folder, 'mask')
    args.rgba_folder = os.path.join(input_folder, 'images')
    args.sparse_folder = os.path.join(input_folder, args.subpath)
    args.undistorted_folder = os.path.join(input_folder, 'undistorted')
    args.database_path = os.path.join(input_folder, args.colmap_db)
    args.output = os.path.join(input_folder, BAND + ".csv")
    
    # Both mask and rgba should be there but if not create them
    if not os.path.exists(args.mask_folder) or not os.path.exists(args.rgba_folder):
        print("{} or {} not found. Please run process.py first to create both.".format(args.mask_folder, args.rgba_folder))
    
    process_video(args)

    check_overwrite(args.output)
    convert_to_csv(args)

    # save metadata
    write_metadata(args.input, data)