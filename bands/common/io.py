import os

import av
import cv2
import numpy as np

from .encode import heat_to_rgb

def create_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def copy_folder(src, dst):
    cmd = "cp -r " + src + " " + dst
    os.system(cmd)


def get_image_size(path):
    img = cv2.imread(path)
    return img.shape[1], img.shape[0]

def get_video_data(path):
    import decord
    video = decord.VideoReader(path)
    return video[0].shape[1], video[0].shape[0], video.get_avg_fps(), len(video)
    

def open_float_rgb(path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    

def to_float_rgb(image):
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def write_rgba(path, rgba):
    cv2.imwrite(path, (rgba * 255).astype(np.uint8))


def write_depth(path, depth, normalize=True, flip=True, heatmap=False):
    """Write depth map to png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    if normalize:
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)

    if flip:
        depth = 1.0-depth

    if heatmap:
        depth = depth.astype(np.float64)
        cv2.imwrite(path, (heat_to_rgb(depth) * 255 ).astype(np.uint8))

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


def make_video(filename, folder=".", fps=24, codec="libx264", pix_fmt="yuv420p", crf=15):
    cmd = "ffmpeg -r " + str( fps ) + " -i " + folder + "/%05d.png -vcodec " + codec + " -crf " + str(crf) + " -pix_fmt " + pix_fmt + " preview.mp4"
    os.system(cmd)

    cmd = "mv preview.mp4 " + filename
    os.system(cmd)


def extract_frames(filename, folder=".", fps=24):
    import decord
    in_video = decord.VideoReader(filename)

    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    fps = in_video.get_avg_fps()
    total_frames = len(in_video)

    if not os.path.exists(folder):
        create_folder(folder)

    # Simple passthrough process to remove audio
    print("Saving video " + output_file)
    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_file)
    for i in tqdm( range(total_frames) ):
        curr_frame = in_video[i].asnumpy()
        write_rgba(output_file, curr_frame)
        out_video.write(curr_frame)
    out_video.close()


class VideoWriter(object):
    def __init__(self, width, height, frame_rate, crf=15, filename="output.mp4"):
        super().__init__()

        # get safe video size (divisible by 2)
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

        # set video parameters
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