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
import argparse

from bands.common.io import get_image_size, get_video_data
from bands.common.meta import create_metadata, is_video, add_band, write_metadata

import warnings
warnings.filterwarnings("ignore")

# Default values
DEPTH_VIDEO_DEFUALT = "depth_zoedepth"
DEPTH_IMAGE_DEFAULT = "depth_patchfusion"
DEPTH_BANDS = ["depth_midas", "depth_marigold", "depth_zoedepth", "depth_patchfusion"]
DEPTH_OPTIONS = DEPTH_BANDS + ["all"]

# Subfolders
SUBFOLDERS = {
    "rgba": "images",
    "mask_mmdet": "mask",
    "flow_raft": "flow",
    "depth_zoedepth": "depth",
    "depth_midas": "depth_midas",
    "depth_marigold": "depth_marigold",
    "depth_patchfusion": "depth_patchfusion",
}


# Run band model
def run(band, input_folder, output_file="", subpath=False, extra_args = ""):
    print("\n# ", band.upper())
    cmd = "CUDA_VISIBLE_DEVICES=0 python3 bands/" + band + ".py -i " + input_folder
    if output_file != "":
        cmd += " --output " + output_file

    if extra_args != "":
        cmd += " " + extra_args

    if subpath:
        cmd += " --subpath " + SUBFOLDERS[band] + " "

    print(cmd,"\n")
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input file", type=str, required=True)
    parser.add_argument('--output', help="folder name", type=str, default='')

    # global video properties
    parser.add_argument('--fps', '-r', help='fix framerate', type=float, default=24)
    parser.add_argument('--extra', '-e', help='Save extra data [>0 frames|PLYs; >1 FLOs; >2 NPY]', type=int, default=0)

    # Depth
    parser.add_argument('--rgbd', help='Where the depth is', type=str, default=None)
    parser.add_argument('--depth', '-d', help='Depth bands', type=str, default=None, choices=DEPTH_OPTIONS)
    parser.add_argument('--ply', '-p', help='Save ply for images', action='store_true')
    parser.add_argument('--npy', '-n', help='Save npy version of files', action='store_true')

    # Flow
    parser.add_argument('--flo', help='Save flo files for raft', action='store_true')
    parser.add_argument('--flow_backwards', '-b',  help="Save backwards video", action='store_true')


    args = parser.parse_args()

    # 1. Get input basic parameters
    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]

    # 2. Create folder
    folder_name = os.path.join(input_folder, input_basename)
    if args.output:
        folder_name = args.output

    data = create_metadata(folder_name)

    if is_video(input_path):
        data["ext"] = "mp4"
    else:
        data["ext"] = input_filename.rsplit( ".", 1 )[ 1 ]

    name_rgba = "rgba." + data["ext"]
    path_rgba = os.path.join(folder_name, name_rgba)
    
    # 3. Extract RGBA (only if doesn't exist)
    add_band(data, "rgba", url=name_rgba)

    extra_args = ""
    if args.rgbd:
        extra_args += "--rgbd " + args.rgbd

    if is_video(input_path):
        extra_args += " --fps " + str(args.fps)

    run("rgba", input_path, path_rgba, subpath=True, extra_args=extra_args)

    # 4. Add metadata
    if is_video(input_path):
        data["width"],  data["height"], data["fps"], data["frames"] = get_video_data(path_rgba)
        data["duration"] = float(data["frames"]) / float(data["fps"])

    else:
        data["width"], data["height"] = get_image_size(path_rgba)

    write_metadata(folder_name, data)
    
    # 5. Extract bands
    # 

    # Set some global properties
    global_args = ""

    if args.extra > 0:
        args.ply = True

    if args.extra > 1:
        args.flo = True
        args.flow_backwards = True

    if args.extra > 2:
        args.npy = True

    # 5.a EXTRACT DEPTH
        
    if args.ply:
        depth_args = "--ply "

    if args.npy:
        depth_args += "--npy "

    # Choose defualt depth band
    if args.depth == None:
        if is_video(input_path):
            args.depth = DEPTH_VIDEO_DEFUALT
        else:
            args.depth = DEPTH_IMAGE_DEFAULT

    # Process depth
    if args.depth == "all":

        for band in DEPTH_BANDS:
            extra_args = depth_args

            if band == "depth_patchfusion" and is_video(input_path):
                extra_args += "--mode=p49 "

            run(band, folder_name, subpath=args.extra, extra_args=extra_args)
    else:
        extra_args = depth_args

        if args.depth == "depth_patchfusion" and is_video(input_path):
            extra_args += "--mode=p49 "

        run(args.depth, folder_name, subpath=args.extra, extra_args=extra_args)
    
    # 5.b EXTRACT MASK (mmdet)
    run("mask_mmdet", folder_name, subpath=True, extra_args="--sdf")

    # 5.c EXTRACT optical FLOW
    if is_video(input_path):
        # Flow (RAFT)
        flow_args = ""
        if args.flo:
            flow_args += "--flo_subpath flow "
        if args.flow_backwards:
            flow_args += "--backwards "
        run("flow_raft", folder_name, subpath=True, extra_args=flow_args)


        



