import os
import argparse

from bands.common.io import get_image_size, get_video_data
from bands.common.meta import create_metadata, is_video, add_band, write_metadata

import warnings
warnings.filterwarnings("ignore")

# Run band model
def run(band, input_folder, output_file="", extra_args = "", force=False):
    print("\n# ", band.upper())
    cmd = "CUDA_VISIBLE_DEVICES=0 python3 bands/" + band + ".py -i " + input_folder
    if output_file != "":
        cmd += " --output " + output_file 
    if extra_args != "":
        cmd += " " + extra_args
    print(cmd,"\n")
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input file", type=str, required=True)
    parser.add_argument('--output', help="folder name", type=str, default='')
    parser.add_argument('--fps', '-r', help='fix framerate', type=float, default=24)
    parser.add_argument('--rgbd', help='Where the depth is', type=str, default='none')
    parser.add_argument('--force', '-f', help='Force to run all bands', action='store_true')
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
    
    add_band(data, "rgba", url=name_rgba)

    # 3. Extract RGBA (only if doesn't exist)
    if is_video(input_path):
        extra_args = "-d images --fps " + str(args.fps)

        if args.rgbd != "none":
            extra_args="--rgbd " + args.rgbd

        run("rgba", input_path, path_rgba, extra_args=extra_args)

        # Add metadata
        data["width"],  data["height"], data["fps"], data["frames"] = get_video_data(path_rgba)
        data["duration"] = float(data["frames"]) / float(data["fps"])

    else:
        input_extension = "png"
        cmd = "cp " + input_path + " " + path_rgba
        os.system(cmd)

        # Add metadata
        data["width"], data["height"] = get_image_size(path_rgba)

    write_metadata(folder_name, data)
    
    # 5. Extract bands

    # Depth (MariGold)
    run("depth_marigold", folder_name)

    # Depth HUE (ZoeDepth w MiDAS v2.1)
    run("depth_zoe", folder_name)

    # Mask (mmdet)
    run("mask_mmdet",  folder_name, extra_args="--sdf")

    if is_video(input_path):
        # Depth (PatchFusion w ZoeDepth)
        run("depth_fusion", folder_name, extra_args="--mode=p49")

        # Flow (RAFT)
        run("flow_raft", folder_name)

    else:        
        # Depth (PatchFusion w ZoeDepth)
        run("depth_fusion", folder_name)

        # Mask inpainting
        run("inpaint_fcfgan", folder_name)

        # scale
        run("scale_realesrgan", folder_name)


        



