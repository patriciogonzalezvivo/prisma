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

# Run band model
def run(band, input_folder, output_file="", save_frames=False, extra_args = ""):
    print("\n# ", band.upper())
    cmd = "CUDA_VISIBLE_DEVICES=0 python3 bands/" + band + ".py -i " + input_folder
    if output_file != "":
        cmd += " --output " + output_file 
    if extra_args != "":
        cmd += " " + extra_args

    if save_frames:
        cmd += " --subpath " + band

    print(cmd,"\n")
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input file", type=str, required=True)
    parser.add_argument('--output', help="folder name", type=str, default='')
    parser.add_argument('--fps', '-r', help='fix framerate', type=float, default=24)
    parser.add_argument('--rgbd', '-d', help='Where the depth is', type=str, default=None)
    parser.add_argument('--frames', '-f', help='Save extra frame data as images', action='store_true')
    parser.add_argument('--ply', '-p', help='Save ply for images', action='store_true')
    parser.add_argument('--npy', '-n', help='Save npy version of files', action='store_true')


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

    extra_args = "--subpath images"
    if args.rgbd:
        extra_args += "--rgbd " + args.rgbd

    if is_video(input_path):
        extra_args += " -d images --fps " + str(args.fps)

    run("rgba", input_path, path_rgba, save_frames=True, extra_args=extra_args)

    # 4. Add metadata
    if is_video(input_path):
        data["width"],  data["height"], data["fps"], data["frames"] = get_video_data(path_rgba)
        data["duration"] = float(data["frames"]) / float(data["fps"])

    else:
        data["width"], data["height"] = get_image_size(path_rgba)

    write_metadata(folder_name, data)
    
    # 5. Extract bands
    # 

    extra_args = ""
    if args.ply:
        extra_args = "--ply "

    if args.npy:
        extra_args += "--npy "


    # Depth (MariGold)
    run("depth_marigold", folder_name, save_frames=args.frames, extra_args=extra_args)

    # Depth HUE (ZoeDepth w MiDAS v2.1)
    run("depth_zoedepth", folder_name, save_frames=args.frames, extra_args=extra_args)

    # Midas v3.1
    run("depth_midas", folder_name, save_frames=args.frames, extra_args=extra_args)

    # Mask (mmdet)
    run("mask_mmdet", folder_name, save_frames=args.frames, extra_args=extra_args + "--sdf")

    if is_video(input_path):
        # Depth (PatchFusion w ZoeDepth)
        run("depth_patchfusion", folder_name, save_frames=args.frames, extra_args=extra_args + " --mode=p49")

        # Flow (RAFT)
        run("flow_raft", folder_name)

    else:        
        # Depth (PatchFusion w ZoeDepth)
        run("depth_patchfusion", folder_name, extra_args=extra_args)


        



