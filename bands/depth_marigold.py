import numpy as np
import torch
import argparse
import os

import warnings
warnings.filterwarnings("ignore")

from marigold import MarigoldPipeline

from common.io import create_folder, check_overwrite, write_depth, write_pcl
from common.meta import load_metadata, get_target, write_metadata, is_video, get_url
from common.encode import heat_to_rgb

from PIL import Image

BAND = "depth_marigold"
CHECKPOINT = "Bingxin/Marigold"
DEVICE_CUDA = 'cuda' if torch.cuda.is_available else 'cpu'
DEVICE_APPLE = 'mps:0' if torch.cuda.is_available else 'cpu'
DENOISE_STEPS = 10
ENSEMBLE_STEPS = 10
PROCESSING_RESOLUTION = 768

device = None
model = None
batch_size = 0
data = None


# Load MariGold pipeline
def init_model(checkpoint=CHECKPOINT, apple_silicon=False, half_precision=False):
    global model, device, batch_size

    if apple_silicon:
        device = torch.device( DEVICE_APPLE )
        if batch_size == 0:
            batch_size = 1 
    else:
        device = torch.device( DEVICE_CUDA )

    if half_precision:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = MarigoldPipeline.from_pretrained(checkpoint, torch_dtype=dtype)

    try:
        import xformers
        model.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    model = model.to(device)
    return model


def infer(img, denoising_steps=DENOISE_STEPS, ensemble_size=ENSEMBLE_STEPS, processing_res=PROCESSING_RESOLUTION, normalize=False):
    global model

    if model == None:
        init_model()

    # Predict depth
    pipe_out = model(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        match_input_res=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    prediction = pipe_out.depth_np
        
    if normalize:
        # Normalization
        depth_min = prediction.min()
        depth_max = prediction.max()

        if depth_max - depth_min > np.finfo("float").eps:
            prediction = (prediction - depth_min) / (depth_max - depth_min)

    return prediction


def process_image(args):
    output_folder = os.path.dirname(args.output)
    in_image = Image.open(args.input).convert("RGB")

    prediction = infer(in_image, normalize=False)

    if data:
        depth_min = prediction.min().item()
        depth_max = prediction.max().item()
        data["bands"][BAND]["values"] = { 
                                                "min" : {
                                                        "value": depth_min, 
                                                        "type": "float"
                                                },
                                                "max" : {
                                                    "value": depth_max,
                                                    "type": "float" }
                                            }
    if args.npy:
        np.save( os.path.join(output_folder, BAND + '.npy'), prediction)

    if args.ply:
        write_pcl( os.path.join(output_folder, BAND + '.ply'), 0.5 + prediction * 2.0, np.array(in_image))

    # Save depth
    write_depth( args.output, prediction, flip=False, heatmap=True, encode_range=True)


def process_video(args):
    import decord
    from tqdm import tqdm
    from common.io import VideoWriter

    # LOAD resource 
    in_video = decord.VideoReader(args.input)
    width = in_video[0].shape[1]
    height = in_video[0].shape[0]
    total_frames = len(in_video)
    fps = in_video.get_avg_fps()

    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )

    output_folder = os.path.dirname(args.output)
    if args.subpath != '':
        if data:
            data["bands"][BAND]["folder"] = args.subpath
        args.subpath = os.path.join(output_folder, args.subpath)
        create_folder(args.subpath)

    csv_files = []
    for i in tqdm( range(total_frames) ):
            
        img = Image.fromarray(np.uint8(in_video[i].asnumpy())).convert('RGB')
        prediction = infer(img, normalize=False)

        if args.npy:
            if args.subpath != '':
                np.save( os.path.join(args.subpath, "{:05d}.npy".format(i)), prediction)
            else:
                np.save(os.path.join(os.path.join(output_folder, BAND + '_npy', prediction), '%04d.npy' % i), prediction)

        depth_min = prediction.min()
        depth_max = prediction.max()
        depth = 1.0 - (prediction - depth_min) / (depth_max - depth_min)
        out_video.write( ( heat_to_rgb(depth.astype(np.float64)) * 255 ).astype(np.uint8) )

        if args.subpath != '':
            write_depth( os.path.join(args.subpath, "{:05d}.png".format(i)), prediction, normalize=False, flip=False, heatmap=True)

        csv_files.append( ( depth_min.item(),
                            depth_max.item()  ) )

    out_video.close()

    csv_min = open( os.path.join( output_folder, BAND + "_min.csv" ) , 'w')
    csv_max = open( os.path.join( output_folder, BAND + "_max.csv" ) , 'w')

    for e in csv_files:
        csv_min.write( '{}\n'.format(e[0]) )
        csv_max.write( '{}\n'.format(e[1]) )

    csv_min.close()
    csv_max.close()

    if data:
        data["bands"][BAND]["values"] = { 
                                            "min" : {
                                                    "type": "float",
                                                    "url": BAND + "_min.csv"
                                            },
                                            "max" : {
                                                "type": "float", 
                                                "url": BAND + "_max.csv",
                                            }
                                        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', help="Input image/video", type=str, required=True)
    parser.add_argument('--output', '-o', help="Output image/video", type=str, default="")
    parser.add_argument('--npy' , '-n', help="Save numpy data", action='store_true')
    parser.add_argument('--ply' , '-p', help="Create point cloud PLY", action='store_true')
    parser.add_argument('--subpath', '-d', help="subpath to frames", type=str, default='')

    parser.add_argument('--checkpoint','-c', help="checkpoint", type=str, default=CHECKPOINT)
    args = parser.parse_args()

    # Try to load metadata
    data = load_metadata(args.input)
    if data:
        # IF the input is a PRISMA folder it can use the metadata defaults
        print("PRISMA metadata found and loaded")
        args.input = get_url(args.input, data, "rgba")
        args.output = get_target(args.input, data, band=BAND, target=args.output, force_image_extension="png")

    # Check if the output folder exists
    check_overwrite(args.output)

    if args.npy and is_video(args.output):
        os.makedirs(os.path.join(os.path.dirname(args.output), BAND + "_npy"), exist_ok=True)

    # compute depth maps
    init_model(checkpoint=args.checkpoint) 
    
    if is_video(args.output):
        process_video(args)
    else:
        process_image(args)

    # save metadata
    write_metadata(args.input, data)