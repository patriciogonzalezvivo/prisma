import os
import json
import argparse

from bands.common.io import get_image_size, get_video_data

import warnings
warnings.filterwarnings("ignore")

# Run band model
def run(band, input_file, output_file="", extra_args = "", force=False):

    if not force and output_file != "":
        if os.path.exists(output_file):
            print("Output file already exists. Skipping...")
            return

    print("\n# ", band.upper())
    cmd = "CUDA_VISIBLE_DEVICES=0 python3 bands/" + band + ".py -i " + input_file
    if output_file != "":
        cmd += " -output " + output_file 
    if extra_args != "":
        cmd += " " + extra_args
    print(cmd,"\n")
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="input file", type=str, required=True)
    parser.add_argument('-output', help="folder name", type=str, default='')
    parser.add_argument('-fps', '-r', help='fix framerate', type=float, default = 0)
    parser.add_argument('-rgbd', help='Where the depth is', type=str, default='none')
    parser.add_argument('-force', '-f', help='Force to run all bands', action='store_true')
    args = parser.parse_args()

    # 1. Get input basic parameters
    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    # 2. Create folder
    folder_name = os.path.join(input_folder, input_basename)
    if args.output:
        folder_name = args.output
    name_rgba = "rgba." + input_extension
    path_rgba = os.path.join(folder_name, name_rgba)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

        # 3. Extract RGBA (only if doesn't exist)
        if input_video:
            input_extension = "mp4"
            extra_args = ""
            if args.rgbd != "none":
                extra_args="-rgbd " + args.rgbd
                run("rgba", input_path, path_rgba, extra_args=extra_args)
        else:
            input_extension = "png"

        cmd = "cp " + input_path + " " + path_rgba
        os.system(cmd)

    # 4. Create payload
    with open( os.path.join(folder_name, "payload.json"), 'w') as payload:
        data = {
            "bands": {
                "rgba": { 
                    "url": name_rgba,
                },
            }
        }

        if args.rgbd != "none":
            data["bands"]["depth"] = {
                "url": "depth.mp4",
            }
        
        if input_video:
            data["width"],  data["height"], data["fps"], data["frames"] = get_video_data(path_rgba)
            data["duration"] = float(data["frames"]) / float(data["fps"])
            data["bands"]["rgba"]["folder"] = "rgba"
            
        else:
            data["width"], data["height"] = get_image_size(path_rgba)

        payload.write( json.dumps(data, indent=4) )

    # 5. Extract bands
    if input_video:
        # Depth (PatchFusion w ZoeDepth)
        run("depth_fusion", folder_name, extra_args="-mode=p49")

        # Depth HUE (ZoeDepth w MiDAS v2.1)
        run("depth_zoe", folder_name)

        # # Depth (MIDAS v3.1)
        # run("depth_midas", folder_name)

        # Flow (RAFT)
        run("flow_raft", folder_name)

        # Mask (detectron2)
        run("mask",  folder_name)

        # DensePose (detectron2)
        run("mask_densepose",  folder_name)

        # Camera estimation using COLMAP
        run("camera_colmap", folder_name)

    else:
        # Depth (PatchFusion w ZoeDepth)
        run("depth_fusion", folder_name)

        # Depth (ZoeDepth w MiDAS v2.1)
        run("depth_zoe", folder_name)

        # # Depth (MIDAS v3.1)
        # run("depth_midas", folder_name)

        # Mask (detectron2)
        run("mask",  folder_name)

        # DensePose (detectron2)
        run("mask_densepose",  folder_name)

        # Pose (mediapipe)
        run("mask_mediapipe",  folder_name)

        # Mask inpainting
        run("inpaint_fcfgan", folder_name)

        # scale
        run("scale_realesrgan", folder_name)


        



