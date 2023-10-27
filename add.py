import os
import json
import argparse

import warnings
warnings.filterwarnings("ignore")

# Run band model
def run(band, input_file, output_file="", extra_args = ""):
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
            "band": {
                "rgba": { 
                    "url": name_rgba, 
                    "folder": "rgba",
                },
            }
        }

        if args.rgbd != "none":
            data["band"]["depth"] = {
                "url": "depth.mp4",
            }
        
        if input_video:
            import decord
            video = decord.VideoReader(path_rgba)
            data["width"] = video[0].shape[1]
            data["height"] = video[0].shape[0]
            data["fps"] = video.get_avg_fps()
            data["frames"] = len(video)
            data["duration"] = float(data["frames"]) / float(data["fps"])
            
        else:
            import cv2
            image = cv2.imread(path_rgba)
            data["width"] = image.shape[1]
            data["height"] = image.shape[0]

        payload.write( json.dumps(data, indent=4) )

    # 5. Extract bands
    if input_video:
        # Depth HUE (ZoeDepth)
        run("depth_zoe", folder_name)
        run("depth_midas3", folder_name)

        # Flow (RAFT)
        run("flow_raft", folder_name)

        # Mask (mmdet)
        run("mask_mmdet",  folder_name)

        # Camera estimation using COLMAP
        run("camera_colmap", folder_name)

    else:
        # Depth (ZoeDepth w MiDAS v2.1)
        run("depth_zoe", folder_name)

        # Depth (MIDAS v3.1)
        run("depth_midas3", folder_name)

        # Mask (mmdet)
        run("mask_mmdet",  folder_name)

        # Mask inpainting
        run("inpaint_fcfgan", folder_name)

        # scale
        run("scale_realesrgan", folder_name)


        



