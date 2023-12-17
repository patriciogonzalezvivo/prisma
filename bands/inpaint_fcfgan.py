import os
import numpy as np
import argparse
import json

import cv2
import torch
import dnnlib

import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import legacy

DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
MODEL  = 'models/places_512.pkl'
# MODEL  = 'models/places.pkl'

device = None
model = None
label = None
data = None

def init_model(class_idx=None, resolution=512):
    global model, device, label

    seed = 0
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(DEVICE)
    
    print('Loading networks from "%s"...' % MODEL)
    with dnnlib.util.open_url(MODEL) as f:
        model = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    model = model.eval().to(device)
    
    # Labels.
    label = torch.zeros([1, model.c_dim], device=device)
    if model.c_dim != 0:
        if class_idx is None:
            print('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    
    netG_params = sum(p.numel() for p in model.parameters())
    print("Generator Params: {} M".format(netG_params/1e6))
    print("Starting Visualization...")
    

def infer(rgb, mask, resolution=256, truncation_psi=0.1):
    global model, device, label

    if model == None:
        init_model()

    width = rgb.shape[1]
    height = rgb.shape[0]

    # uint8
    rgb = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    rgb = rgb.transpose(2,0,1)
    rgb = torch.from_numpy(rgb.astype(np.float32))
    rgb = (rgb.to(torch.float32) / 127.5 - 1)

    mask = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask).to(torch.float32)
    mask_tensor = mask_tensor.unsqueeze(0)

    rgb_erased = rgb.clone()
    rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
    rgb_erased = rgb_erased.to(torch.float32)

    with torch.no_grad():
        erased_img = torch.stack(list([rgb_erased]), dim=0).to(device)
        mask = torch.stack(list([mask_tensor]), dim=0).to(device)

        pred_img = model(   img=torch.cat([0.5 - mask, erased_img], dim=1), 
                            c=label, 
                            truncation_psi=truncation_psi, 
                            noise_mode='const')
        
        comp_img = mask_tensor.to(device) * pred_img + (1 - mask_tensor).to(device) * rgb.to(device)

        lo, hi = [-1, 1]
        comp_img = comp_img.detach()
        comp_img = np.asarray(comp_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
        comp_img = (comp_img - lo) * (255 / (hi - lo))
        comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        comp_img = cv2.resize(comp_img, (width, height), interpolation=cv2.INTER_CUBIC)


        return comp_img


def process_image(args):
    output_path = args.output
    output_filename = os.path.basename(output_path)

    rgb = np.array(Image.open(args.input).convert('RGB')) 

    mask = np.array(Image.open(args.mask).convert('RGB').convert('L')) / 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = infer(rgb, mask, args.resolution)

    cv2.imwrite(args.output, inpainted.astype(np.uint8))

    if data:
        data["bands"]["mask_inpaint"] = { }
        data["bands"]["mask_inpaint"]["url"] = output_filename


def process_video(args):
    import decord
    from tqdm import tqdm
    from common.io import VideoWriter

    print("load video", args.input, args.mask)
    output_path = args.output
    print("save as", output_path)
    output_filename = os.path.basename(output_path)

    # LOAD resource 
    rgba_video = decord.VideoReader(args.input)
    mask_video = decord.VideoReader(args.mask)
    width = rgba_video[0].shape[1]
    height = rgba_video[0].shape[0]
    total_frames = len(rgba_video)
    fps = rgba_video.get_avg_fps()

    inpainted_video = VideoWriter(width=width, height=height, frame_rate=fps, filename=output_path)

    for f in tqdm( range(total_frames) ):
        rgba = cv2.cvtColor(rgba_video[f].asnumpy(), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask_video[f].asnumpy(), cv2.COLOR_BGR2GRAY).astype(np.uint8)

        mask = np.array(mask) / 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        inpainted = infer(rgba, mask, args.resolution)
        inpainted_video.write( inpainted.astype(np.uint8) )

    inpainted_video.close()

    if data:
        data["bands"]["mask_inpaint"] = { }
        data["bands"]["mask_inpaint"]["url"] = output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input', '-i', help="input image", type=str, required=True)
    parser.add_argument('-mask', '-m', help="input mask", type=str, default="")
    parser.add_argument('-output', '-o', help="output", type=str, default="")
    parser.add_argument('-resolution', '-r', help="resolution", type=int, default=512)

    args = parser.parse_args()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)
    input_payload = os.path.join( input_folder, "payload.json")
    input_filename = os.path.basename(input_path)
    input_basename = input_filename.rsplit(".", 1)[0]
    input_extension = input_filename.rsplit(".", 1)[1]
    input_video = input_extension == "mp4"

    if args.mask == "":
        args.mask = os.path.join( input_folder, data["bands"]["mask"]["url"] )

    if not input_video:
        input_extension = "png"

    if os.path.isdir( args.output ):
        args.output = os.path.join(args.output, "mask_inpaint." + input_extension)
    elif args.output == "":
        args.output = os.path.join(input_folder, "mask_inpaint." + input_extension)

    if os.path.isfile(input_payload):
        if not data:
            data = json.load( open(input_payload) )


    if not input_video:
        process_image(args)
    else:
        process_video(args)

    if data:
        with open( input_payload, 'w') as payload:
            payload.write( json.dumps(data, indent=4) )





