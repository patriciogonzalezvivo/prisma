import os
import argparse

import decord
from tqdm import tqdm

from common.io import create_folder, write_rgba
from common.io import VideoWriter

BAND = "rgba"

def split(input_file, output_file, fps, total_frames, width, height, split):
    print("TODO")


def run(args):
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)

    if args.rgbd == "none":
        # prune(args.input, args.output)

        # Simple passthrough process to remove audio
        print("Saving video " + args.output)
        out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output)
        for i in tqdm( range(total_frames) ):
            curr_frame = in_video[i].asnumpy()
            out_video.write(curr_frame)
        out_video.close()

    else:
        rgb = "none"
        if args.rgbd == "right":
            rgb = "left"
        elif args.rgbd == "left":
            rgb = "right"
        elif args.rgbd == "top":
            rgb = "bottom"
        elif args.rgbd == "bottom":
            rgb = "top"
        split(args.input, args.output, total_frames, width, height, split=rgb)
        # prune(args.output, args.output)
        split(args.input, args.depth, total_frames, width, height, split=args.rgbd)
        # prune(args.depth, args.depth)

    # remove tmp file
    if os.path.exists(args.tmp):
        os.remove(args.tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-tmp', '-t', help="tmp", type=str, default="tmp.mp4")
    parser.add_argument('-output', help="output", type=str, default="")
    parser.add_argument('-rgbd', help='Where the depth is', type=str, default='none')
    parser.add_argument('-depth', help='in case of being a RGBD', type=str, default="depth_hue.mp4")
    parser.add_argument('-fps', '-r', help='fix framerate', type=float, default=24)
    args = parser.parse_args()

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + ".mp4")
    args.depth = os.path.join(os.path.dirname(args.output), args.depth)
    
    run(args)
