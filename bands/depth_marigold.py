import numpy as np
import torch
import json
import argparse
import os
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from marigold import MarigoldPipeline
from marigold.util.seed_all import seed_all

from common.encode import heat_to_rgb
from common.io import create_folder, write_depth

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


def infer(img, denoising_steps=DENOISE_STEPS, ensemble_size=ENSEMBLE_STEPS, processing_res=PROCESSING_RESOLUTION, normalize=True):
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

    output_folder = os.path.dirname(args.output)
    output_folder = os.path.join(output_folder, BAND)
    create_folder(output_folder)
    if data:
        data["bands"][BAND]["folder"] = BAND

    out_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=args.output )

    csv_files = []
    for i in tqdm( range(total_frames) ):
            
        img = Image.fromarray(np.uint8(in_video[i].asnumpy())).convert('RGB')
        prediction = infer(img, normalize=False)

        if args.npy:
            output_folder = os.path.dirname(args.output)
            np.save(os.path.join(os.path.join(output_folder, BAND + '_npy', prediction), '%04d.npy' % i), prediction)

        depth_min = prediction.min()
        depth_max = prediction.max()

        depth = 1.0 - (prediction - depth_min) / (depth_max - depth_min)

        out_video.write( ( heat_to_rgb(depth.astype(np.float64)) * 255 ).astype(np.uint8) )

        write_depth( os.path.join(output_folder, "{:05d}.png".format(i)), prediction, normalize=False, flip=False, heatmap=True)

        csv_files.append( ( depth_min.item(),
                            depth_max.item()  ) )

    out_video.close()

    output_folder = os.path.dirname(args.output)
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

def process_image(args):
    img = Image.open(args.input).convert("RGB")
    prediction = infer(img, normalize=False)

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
        output_folder = os.path.dirname(args.output)
        np.save( os.path.join(output_folder, BAND + '.npy', prediction) )

    # Save depth
    write_depth( args.output, prediction, flip=False, heatmap=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input", type=str, required=True)
    parser.add_argument('-output', '-o', help="output", type=str, default="")
    parser.add_argument('-checkpoint', help="checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument('-npy' , '-n', help="Keep numpy data", action='store_true')
    args = parser.parse_args()

    init_model(checkpoint=args.checkpoint) 

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )
        
    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join(input_folder, "payload.json")
    if os.path.isfile(input_payload):
        data = json.load( open(input_payload) )
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit( ".", 1 )[ 0 ]
    input_extension = input_filename.rsplit( ".", 1 )[ 1 ]
    input_video = input_extension == "mp4"

    if not input_video:
        input_extension = "png"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, BAND + "." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, BAND + "." + input_extension)

    print("output", args.output)
    output_path = args.output
    output_folder = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    output_basename = output_filename.rsplit(".", 1)[0]
    output_extension = output_filename.rsplit(".", 1)[1]

    if args.npy and input_video:
        os.makedirs(os.path.join(output_folder, BAND + "_npy"), exist_ok=True)

    if data:
        data["bands"][BAND] = { "url": output_filename }

    # compute depth maps
    if input_video:
        process_video(args)
    else:
        process_image(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )