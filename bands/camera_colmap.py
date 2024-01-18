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
import sys
import subprocess

import shutil
import argparse
import numpy as np

from common.meta import load_metadata, is_video, get_url, get_target, write_metadata
from common.io import check_overwrite, create_folder, get_check_overwrite

BAND = "camera_pose"
data = None

def run_colmap(args):
    create_folder(args.sparse_folder)

    # Feature extraction
    if get_check_overwrite(args.sparse_folder):

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

        # Reconstruct sparse
        try:
            subprocess.run(mapper_command, check=True)
        except subprocess.CalledProcessError:
            print("mapper failed. relaxing search for initial pair.")
            mapper_command += [
                "--Mapper.init_max_forward_motion", str(1.0),
                "--Mapper.init_min_triangle", str(4),
            ]
            subprocess.run(mapper_command, check=True)
        
    # Bundle adjustment
    if args.colmap_refine:
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

    # # if there are no images on sparse folder, copy rgba images
    # if len(os.listdir(args.sparse_folder)) == 1:
    #     # copy rgba images to sparse folder
    #     for f in os.listdir(args.rgba_folder):
    #         shutil.copy(os.path.join(args.rgba_folder, f), os.path.join(args.sparse_folder, f))


def convert_to_csv(args, outpath):
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
    print(cam, params)

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

    np.savetxt(outpath, lines, delimiter=",", newline="\n",
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
    parser.add_argument('--subpath', help="Use subpath", action='store_true')
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
    args.undistorted_folder = os.path.join(input_folder, 'undistorted')
    args.sparse_folder = os.path.join(input_folder, 'sparse')
    args.database_path = os.path.join(input_folder, args.colmap_db)
    csv_path = os.path.join(input_folder, "camera_pose.csv")

    check_overwrite(csv_path)
    
    # Both mask and rgba should be there but if not create them
    if not os.path.exists(args.mask_folder) or not os.path.exists(args.rgba_folder):
        print("{} or {} not found. Please run process.py first to create both.".format(args.mask_folder, args.rgba_folder))
    
    run_colmap(args)

    convert_to_csv(args, csv_path)



    # save metadata
    write_metadata(args.input, data)