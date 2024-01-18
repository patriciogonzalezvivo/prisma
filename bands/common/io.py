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

import av
import cv2
import numpy as np
from PIL import Image

from .encode import heat_to_rgb, float_to_rgb, float_to_edge, saturation
from .geom import create_point_cloud, save_point_cloud


#
# FS FUNCTIONS
#

def create_folder(dir):
    """Create a folder if it does not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def copy_folder(src, dst):
    """Copy a folder if it does not exist."""
    cmd = "cp -r " + src + " " + dst
    os.system(cmd)


def check_overwrite(path):
    """Check if a file exists and ask the user if it can be overwritten."""
    if os.path.exists(path):
        print("File exists: " + path)
        answer = input("Do you want to over write it? [y/N]: ") 
        if answer != "y":
            exit()

def get_check_overwrite(path):
    """Check if a file exists and ask the user if it can be overwritten."""
    if os.path.exists(path):
        print("File exists: " + path)
        answer = input("Do you want to over write it? [y/N]: ") 
        return answer != "y"
    
    return True


#
# MEDIA INFO FUNCTIONS
#
def get_image_size(path):
    """Get image size, as a tuple (width, height)."""
    img = cv2.imread(path)
    return img.shape[1], img.shape[0]


def get_video_data(path):
    """Get video data, as a tuple (width, height, fps, total_frames)."""
    import decord
    video = decord.VideoReader(path)
    return video[0].shape[1], video[0].shape[0], video.get_avg_fps(), len(video)
    
#
# OPEN FUNCTIONS
# 
def open_float_rgb(path):
    """Open image as float RGB with a range between 0.0 and 1.0."""
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0


def open_rgb(path):
    """Open image as RGB with a range between 0 and 255."""
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def open_image(path):
    """Open image as PIL Image."""
    return Image.open(path).convert("RGB")
    
#
# CONVERT FUNCTIONS
#

def to_float_rgb(image):
    """Convert image to float RGB with a range between 0.0 and 1.0."""
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def to_image(numpy_array):
    """Convert numpy array to PIL Image."""
    return Image.fromarray( np.uint8( numpy_array ) ).convert('RGB')

# 
# WRITE FUNCTIONS
# 

def write_rgb(path, rgb):
    """Write RGB image to png file."""
    rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, rgb)
    
    # # check if rgb is an instance of PIL Image
    # if isinstance(rgb, Image.Image):
    #     rgb.save(path)
    # else:
    #     result = Image.fromarray((rgb * 255).astype(np.uint8))
    #     result.save(path)


# Make image squared of a specific resolution by adding padding into the smaller side 
def write_rgb_square(path, rgb, resolution=1024):
    """Scales and pads a RGB image into a square image."""
    if rgb.shape[0] > rgb.shape[1]:
        pad = (rgb.shape[0] - rgb.shape[1]) // 2
        rgb = np.pad(rgb, ((0, 0), (pad, pad), (0, 0)), mode='constant', constant_values=0)
        
    elif rgb.shape[0] < rgb.shape[1]:
        pad = (rgb.shape[1] - rgb.shape[0]) // 2
        rgb = np.pad(rgb, ((pad, pad), (0, 0), (0, 0)), mode='constant', constant_values=0)

    rgb = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def write_depth(path, depth, normalize=True, flip=False, heatmap=False, encode_range=True):
    """Write depth map to an image file, Either as a heatmap or as a 16-bit png. 
    Heatmaps by default contain the min and max depth ranges encoded in the first two pixels."""
    if normalize:
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)

    if flip:
        depth = 1.0-depth

    if heatmap:
        edge = float_to_edge(depth, ksize=1)
        depth = depth.astype(np.float64)
        
        # encode depth into a heatmap pattern (blue close, red far)
        rgb = heat_to_rgb(depth)

        # encode edge into the saturation channel
        rgb = saturation(rgb, 1.0 - edge)

        # encode min and max depth in the image in the first two pixels
        if encode_range:
            rgb[0,0] = float_to_rgb(depth_min, 0.0, 1000.0)
            rgb[0,1] = float_to_rgb(depth_max, 0.0, 1000.0)

        rgb = (rgb * 255 ).astype(np.uint8)

        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    else:
        bits = 2
        max_val = (2**(8*bits))-1
        depth = depth * max_val
        cv2.imwrite(path, depth.astype("uint16"))


def write_flow(flow, filename):
    """
    Write optical flow in Middlebury .flo format
    
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    
    from https://github.com/liruoteng/OpticalFlowToolkit/
    
    """
    # forcing conversion to float32 precision
    flow = flow.cpu().data.numpy()
    flow = flow.astype(np.float32)
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def write_pcl(filename, depth, rgb, flip=False):
    """Write point cloud to a ply file."""
    if flip:
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = 1.0-depth
        depth = depth_min + depth * (depth_max - depth_min)

    pcl = create_point_cloud(depth, rgb.shape[1]/2, rgb.shape[0]/2)
    save_point_cloud(pcl.reshape((-1, 3)), rgb.reshape(-1, 3), filename)


def extract_frames_from_video(video_path, output, extension="jpg", invert=None, fps=None):
    import subprocess
    
    outpattern = os.path.join(output, "%03d." + extension)

    vf = "yadif"
    if fps is not None:
        vf = f"{vf},fps={fps}"

    if invert is not None:
        vf = f"{vf},negate"

    command = ["ffmpeg",
               "-y",
               "-i", video_path,
               "-q:v", str(1),
               "-vf", vf,
               outpattern]
    print(command)

    subprocess.run(command)


def make_video(filename, folder=".", fps=24, codec="libx264", pix_fmt="yuv420p", crf=15):
    """Make a video from a folder of images."""
    cmd = "ffmpeg -r " + str( fps ) + " -i " + folder + "/%05d.png -vcodec " + codec + " -crf " + str(crf) + " -pix_fmt " + pix_fmt + " preview.mp4"
    os.system(cmd)

    cmd = "mv preview.mp4 " + filename
    os.system(cmd)


class VideoWriter(object):
    """Video writer class."""

    def __init__(self, width, height, frame_rate, crf=15, filename="output.mp4"):
        super().__init__()

        max_size = 3840
        video_width = width
        video_height = height
        if width > max_size or height > max_size:
            aspect_ratio = height / width
            if aspect_ratio < 1:
                video_width = max_size
                video_height = round(max_size * aspect_ratio)
            else:
                video_width = round(max_size / aspect_ratio)
                video_height = max_size
        video_width = 2 * round(video_width / 2)
        video_height = 2 * round(video_height / 2)

        self.options = {
            "width": video_width,
            "height": video_height,
            "frame_rate": frame_rate,
            "crf": crf,
            "codec": "libx264",
            "pix_fmt": "yuv420p",
        }

        self.container = av.open(filename, mode="w")
        self.stream = self.container.add_stream(self.options["codec"], rate="%.2f" % frame_rate)
        self.stream.width = video_width
        self.stream.height = video_height
        self.stream.pix_fmt = self.options["pix_fmt"]
        self.stream.thread_type = "AUTO"
        self.stream.options = {}
        self.stream.options["crf"] = str(self.options["crf"])


    def write(self, frame: np.array, format=None):
        if format == None:        
            if len(frame.shape) == 2:
                format = "gray"
            elif frame.shape[2] == 1:
                format = "gray"
            elif frame.shape[2] == 3:
                format = "rgb24"
            elif frame.shape[2] == 4:
                format = "rgba32"

        frame = av.VideoFrame.from_ndarray(frame.astype(np.uint8), format=format)
        reformatted_frame = frame.reformat(width=self.options["width"], height=self.options["height"])
        packet = self.stream.encode(reformatted_frame)
        self.container.mux(packet)


    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()