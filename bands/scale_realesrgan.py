import argparse
import cv2
import os
import json

import warnings
warnings.filterwarnings("ignore")

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# import time

model = None
scale = 4
window_size = 8
file_url = None

def init_model(name):
    global model, device, scale, file_url

    # determine models according to model names
    if name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        scale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        scale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        scale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        scale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        scale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    model_path = os.path.join('../models', name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, '../models'), progress=True, file_name=None)

    # restorer
    model = RealESRGANer(
        scale=scale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None)

    return model;


def infer(img):
    global model, scale
    output, _ = model.enhance(img, outscale=scale)
    return output


def process_image(args, data = None):
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)

    try:
        output = infer(img)

    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        cv2.imwrite(args.output, output)


def process_video(args, data = None):
    import decord
    from tqdm import tqdm
    from common.io import VideoWriter

    # LOAD resource 
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()

    width *= 2
    height *= 2

    output_path = args.output
    scaled_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path )

    for f in tqdm( range(total_frames) ):
        img = Image.fromarray(in_video[f].asnumpy())
        scaled = infer(img, args.scale)
        scaled_video.write( (scaled * 255).astype(np.uint8) )

    scaled_video.close()


def main():
    global scale
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  help="input image", type=str, required=True)
    parser.add_argument('-o', '--output', help="output", type=str, default="")
    parser.add_argument('-n','--model_name', type=str, default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-s', '--scale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    args = parser.parse_args()

    data = None
    args.model_name = args.model_name.split('.')[0]
    if args.scale == 2:
        args.model_name = "RealESRGAN_x2plus"

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["band"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join( input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit(".", 1)[0]
    input_extension = input_filename.rsplit(".", 1)[1]
    input_video = input_extension == "mp4"

    if not input_video:
        input_extension = "jpg"

    name = input_basename + "_scaled"
    output_filename = name + "." + input_extension


    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, output_filename)
    elif args.output == "":
        args.output = os.path.join(input_folder, output_filename)

    if os.path.isfile(input_payload):  
        if not data:
            data = json.load( open(input_payload) )

    init_model(args.model_name)

    if input_video:
        process_video(args, data)
    else:
        process_image(args, data)

    if data:
        data["band"][name] = { }
        data["band"][name]["url"] = output_filename
        data["band"][name]["scale"] = int(scale)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )
    


if __name__ == '__main__':
    main()
