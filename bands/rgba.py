# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#


import os
import argparse

import decord
from tqdm import tqdm

import numpy as np

from common.io import open_float_rgb, check_overwrite, write_rgb, VideoWriter
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import rgb_to_hsv, heat_to_rgb

BAND = "rgba"

data = None

def split(in_video, output_rgb_file, output_depth_file, split, fps=24, subpath_rgb=None, subpath_depth=None):
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)

    if split == "left":
        rgb_crop = [width/2, 0, width/2, height]
        depth_crop = [0, 0, width/2, height]
    elif split == "right":
        rgb_crop = [0, 0, width/2, height]
        depth_crop = [width/2, 0, width/2, height]
    elif split == "top":
        rgb_crop = [0, height/2, width, height/2]
        depth_crop = [0, 0, width, height/2]
    elif split == "bottom":
        rgb_crop = [0, 0, width, height/2]
        depth_crop = [0, height/2, width, height/2]

    if subpath_rgb:
        output_rgb_folder = os.path.dirname(output_rgb_file)
        subpath_rgb = os.path.join(output_rgb_folder, subpath_rgb)
        if not os.path.exists(subpath_rgb):
            os.makedirs(subpath_rgb)

    if subpath_depth:
        output_depth_folder = os.path.dirname(output_depth_file)
        subpath_depth = os.path.join(output_depth_folder, subpath_depth)
        if not os.path.exists(subpath_depth):
            os.makedirs(subpath_depth)

    rgb_video = VideoWriter(width=rgb_crop[2], height=rgb_crop[3], frame_rate=fps, filename=output_rgb_file)
    depth_video = VideoWriter(width=depth_crop[2], height=depth_crop[3], frame_rate=fps, filename=output_depth_file)
    for i in tqdm( range(total_frames) ):
        curr_frame = in_video[i].asnumpy()
        curr_frame_rgb = curr_frame[int(rgb_crop[1]):int(rgb_crop[1]+rgb_crop[3]), int(rgb_crop[0]):int(rgb_crop[0]+rgb_crop[2]), :]
        curr_frame_depth = curr_frame[int(depth_crop[1]):int(depth_crop[1]+depth_crop[3]), int(depth_crop[0]):int(depth_crop[0]+depth_crop[2]), :]

        if args.encoding_depth == "hue":
            curr_frame_depth = np.clip( rgb_to_hsv(curr_frame_depth)[...,0] / 360.0, 0.0, 1.0)
            curr_frame_depth = heat_to_rgb(curr_frame_depth) * 255.0

        if subpath_rgb:
            write_rgb(os.path.join(subpath_rgb, str(i).zfill(6) + ".png"), 255.0 - np.clip(curr_frame_rgb, 0.0, 255.0))

        if subpath_depth:
            write_rgb(os.path.join(subpath_depth, str(i).zfill(6) + ".png"), 255.0 - np.clip(curr_frame_depth, 0.0, 255.0))

        rgb_video.write(curr_frame_rgb)
        depth_video.write(curr_frame_depth)
        
    rgb_video.close()
    depth_video.close()


def prune(input_file, output_file, fps=24, subpath=None):
    in_video = decord.VideoReader(input_file)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)

    if subpath:
        output_folder = os.path.dirname(output_file)
        subpath = os.path.join(output_folder, subpath)
        if not os.path.exists(subpath):
            os.makedirs(subpath)

    # Simple passthrough process to remove audio
    print("Saving video " + output_file)
    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_file)
    for i in tqdm( range(total_frames) ):
        curr_frame = in_video[i].asnumpy()

        if subpath:
            write_rgb(os.path.join(subpath, str(i).zfill(6) + ".png"), 255.0 - np.clip(curr_frame, 0.0, 255.0))

        out_video.write(curr_frame)
    out_video.close()


def process_image(args):
    print("cp " + args.input + " " + args.tmp)
    os.system("cp " + args.input + " " + args.tmp)

    print("Open", args.tmp, "and save it to", args.output)
    image = open_float_rgb(args.tmp)
    write_rgb(args.output, image)


def process_video(args):
    fps = int(args.fps)

    # use ffmpeg to change fps to 24
    # os.system("ffmpeg -y -i " + args.input + " -filter:v fps=fps=" + str(fps) + " -b:v 10M -maxrate:v 10M -bufsize:v 20M -codec:a copy " + args.tmp)
    # args.input = args.tmp
    os.system("cp " + args.input + " " + args.tmp)

    in_video = decord.VideoReader(args.input)

    if args.rgbd == "none":
        os.system("cp " + args.input + " " + args.tmp)
        prune(args.tmp, args.output, args.fps, args.subpath)

    else:
        split(in_video, output_rgb_file=args.output, output_depth_file=args.output_depth, fps=fps, split=args.rgbd, subpath_rgb=args.subpath)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="input", type=str, required=True)
    parser.add_argument('--tmp', '-t', help="tmp", type=str, default="tmp")
    parser.add_argument('--fps', '-r', help='fix framerate of videos', type=float, default=24)

    # RGB
    parser.add_argument('--output', '-o', help="output", type=str, default="")
    parser.add_argument('--subpath', help="subpath to frames", type=str, default=None)
    
    # Depth
    parser.add_argument('--rgbd', help='Where the depth is', choices=["none", "left", "right", "top", "bottom"], default='none')
    parser.add_argument('--encoding_depth', help="encoding for depth", choices=["none", "hue"], default="none")
    parser.add_argument('--output_depth', help="output file for depth", type=str, default="depth")
    parser.add_argument('--subpath_depth', help="subpath to frames for depth", type=str, default=None)

    args = parser.parse_args()

    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.tmp = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output, force_extension='png')
        if args.rgbd:
            args.output_depth = get_target(args.input, data, band='depth', target=args.output_depth)
    else:
        input_folder = os.path.dirname(args.input)
        input_basename = os.path.basename(args.input)
        input_name = args.input.rsplit( ".", 1 )[ 0 ]
        input_extension = args.input.rsplit( ".", 1 )[ 1 ]

        if args.tmp == "tmp":
            args.tmp = os.path.join(input_folder, "tmp." + input_extension)

        if input_extension != "mp4":
            input_extension = "png"

        if os.path.isdir( args.output ):
            args.output = os.path.join(args.output, BAND + "." + input_extension)
            
        args.output_depth = os.path.join(os.path.dirname(args.output), args.output_depth + "." + input_extension)

    # Check if the output folder exists
    check_overwrite(args.output)
    if args.rgbd:
        check_overwrite(args.output_depth)

    if is_video(args.input):
        process_video(args)
    else:
        process_image(args)

    # remove tmp file
    if os.path.exists(args.tmp):
        os.remove(args.tmp)

    # save metadata
    write_metadata(args.input, data)
