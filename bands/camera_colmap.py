import os
import sys
import json
import subprocess

import shutil
import argparse
import numpy as np

def get_distro():
    import platform
    """
    Name of your Linux distro (in lowercase).
    """
    OS = platform.system()
    if OS == "Linux":
        with open("/etc/issue") as f:
            return f.read().lower().split()[0]
    else:
        return OS
    

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_frames_from_video(video_path, output, extension="jpg", invert=None, fps=None):
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


def extract_frames(args):
    import warnings
    warnings.filterwarnings("ignore")

    import imageio
    import torch
    import torchvision
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    threshold = 0.5

    imgs = []
    reader = imageio.get_reader(args.input)
    for i, im in enumerate(reader):
        imgs.append(im)

    imgs = np.array(imgs)
    num_frames, H, W, _ = imgs.shape
    skip = int(np.ceil(num_frames / 100))
    skip = 1
    imgs = imgs[::skip]

    args.mask_folder = os.path.join(args.output, 'mask')
    args.rgba_folder = os.path.join(args.output, 'rgba')

    create_dir(args.mask_folder)
    create_dir(args.rgba_folder)

    for idx, img in tqdm( enumerate(imgs) ):
        # print(idx)
        imageio.imwrite(os.path.join(args.rgba_folder, str(idx).zfill(3) + '.jpg'), img)

        # Get coarse background mask
        img = torchvision.transforms.functional.to_tensor(img).to(device)
        background_mask = torch.FloatTensor(H, W).fill_(1.0).to(device)
        objPredictions = Maskrcnn([img])[0]

        for intMask in range(len(objPredictions['masks'])):
            if objPredictions['scores'][intMask].item() > threshold:
                if objPredictions['labels'][intMask].item() == 1: # person
                    background_mask[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

        background_mask_np = ((background_mask.cpu().numpy() > 0.1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(args.mask_folder, str(idx).zfill(3) + '.png'), background_mask_np)


def run_colmap(args):
    create_dir(args.sparse_folder)

    # Extract features
    feature_extractor_command = [
        "colmap", "feature_extractor",
        "--database_path", args.database_path,
        "--image_path", args.rgba_folder,
        "--ImageReader.camera_model", args.colmap_camera_model,
        "--ImageReader.single_camera", str(1),
        # "--ImageReader.camera_params", args.colmap_camera_params,
        "--SiftExtraction.use_gpu", str(1),
        # "--SiftExtraction.estimate_affine_shape=true",
        # "--SiftExtraction.domain_size_pooling=true",
        "--SiftExtraction.first_octave", str(0),
    ]

    if args.mask_folder:
        feature_extractor_command += [
            "--ImageReader.mask_path", args.mask_folder,
        ]
    subprocess.run(feature_extractor_command)

    # Match features
    matcher_command = [
        "colmap", 
        args.colmap_matcher + "_matcher",
        "--database_path", args.database_path,
        "--SiftMatching.use_gpu", str(1),
        "--SiftMatching.guided_matching=true",
    ]
    subprocess.run(matcher_command)

    mapper_command = [
        "colmap", "mapper",
        "--database_path", args.database_path,
        "--image_path", args.rgba_folder,
        "--output_path", args.sparse_folder,
        "--Mapper.multiple_models", str(0), 
        "--Mapper.num_threads", str(16),
        "--Mapper.init_min_tri_angle", str(4),
        "--Mapper.extract_colors", str(1),
        "--Mapper.ba_local_max_refinements", str(1),
        "--Mapper.ba_global_max_refinements", str(1),
        "--Mapper.ba_global_function_tolerance=0.000001",
    ]

    # Reconstruct sparse
    if os.path.exists(args.sparse_folder):
        subprocess.run(mapper_command)

        # try:
        #     subprocess.run(mapper_command, check=True)
        # except subprocess.CalledProcessError:
        #     print("mapper failed. relaxing search for initial pair.")
        #     mapper_command += [
        #         "--Mapper.init_max_forward_motion", str(1.0),
        #         "--Mapper.init_min_triangle", str(4),
        #     ]
        #     subprocess.run(mapper_command, check=True)
        
        # Bundle adjustment
        # if args.colmap_refine:
        bundel_adjuster_command = [
            "colmap",
            "bundle_adjuster",
            "--input_path", args.sparse_folder + "/0",
            "--output_path", args.sparse_folder + "/0",
            "--BundleAdjustment.refine_principal_point", str(1)
        ]
        subprocess.run(bundel_adjuster_command)

        # Undistort images (needed for Gaussian Splatting)
        if args.colmap_undistort:
            image_undistorter_command = [
                "colmap", "image_undistorter",
                "--image_path", args.rgba_folder,
                "--input_path", args.sparse_folder + "/0",
                "--output_path", args.undistorted_folder,
                "--output_type", "COLMAP",
                # "--max_image_size", str(1500),
            ]
            subprocess.run(image_undistorter_command)

            files = os.listdir(args.undistorted_folder)
            for file in files:
                if file == '0':
                    continue
                source_file = os.path.join(args.undistorted_folder, file)
                destination_file = os.path.join(args.sparse_folder + "/0", file)
                shutil.move(source_file, destination_file)

        # Convert to txt
        model_converter_command = [
            "colmap", "model_converter",
            "--input_path", args.sparse_folder + "/0",
            "--output_path", args.sparse_folder + "/0",
            "--output_type", "TXT"
        ]
        subprocess.run(model_converter_command)

    # if there are no images on sparse folder, copy rgba images

    if len(os.listdir(args.sparse_folder)) == 1:
        # copy rgba images to sparse folder
        for f in os.listdir(args.rgba_folder):
            shutil.copy(os.path.join(args.rgba_folder, f), os.path.join(args.sparse_folder, f))


def convert_to_csv(sparsedir, outpath, expected_N=None):
    if os.path.exists(os.path.join(sparsedir, "0")):
        sparsedir = os.path.join(sparsedir, "0")

    sys.path.append(".")
    from bands.common.colmap import read_model
    import numpy as np
    cameras, model_images, points3D = read_model(path=sparsedir)

    keys = list(model_images.keys())
    keys = sorted(keys, key=lambda x: model_images[x].name)

    print(len(keys))
    if expected_N is not None:
        assert len(keys) == expected_N

    camkey = model_images[keys[0]].camera_id
    # for key in keys:
    #     print(model_images[key].camera_id)
        # assume single camera setup since we are dealing with videos
        # assert model_images[key].camera_id == camkey

    cam = cameras[camkey]
    params = cam.params
    print(cam, params)

    # assert cam.model in ["RADIAL", "SIMPLE_RADIAL"]
    K = np.array([[params[0], 0.0, params[1]],
                  [0.0, params[0], params[2]],
                  [0.0,       0.0,       1.0]])

    Rs = np.stack([model_images[k].qvec2rotmat() for k in keys])
    ts = np.stack([model_images[k].tvec for k in keys])

    N = Rs.shape[0]
    params = params[:3][None].repeat(N, axis=0)
    Rs = Rs.reshape(N, 9)

    lines = np.concatenate((params, Rs, ts), axis=1)

    np.savetxt(outpath, lines, delimiter=",", newline="\n",
               header=(",".join(["f", "ox", "oy"]+
                                [f"R[{i//3},{i%3}]" for i in range(9)]+
                                [f"t[{i}]" for i in range(3)])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '-i', type=str, help='input video path', required=True)
    parser.add_argument('-output', '-o', type=str, help='where to store data', default='')
    parser.add_argument('-force', '-f', action="store_true", help="Force re-computation of all results.")
    parser.add_argument('-colmap_db', default="colmap.db", help="colmap database filename")
    parser.add_argument('-colmap_camera_model', default="SIMPLE_PINHOLE", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
    parser.add_argument('-colmap_camera_params', default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
    parser.add_argument('-colmap_matcher', default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
    # parser.add_argument('-colmap_refine','-r',  help="adjust bundle by refining cameras", action='store_true')
    parser.add_argument('-colmap_undistort','-u',  help="undistort images", action='store_true')
    args = parser.parse_args()

    if os.path.isdir( args.input ):
        payload_path = os.path.join( args.input, "payload.json")
        if os.path.isfile(payload_path):
            data = json.load( open(payload_path) )
            args.input = os.path.join( args.input, data["bands"]["rgba"]["url"] )

    input_path = args.input
    input_folder = os.path.dirname(input_path)

    if args.output == "":
        args.output = input_folder

    args.mask_folder = os.path.join(args.output, 'mask')
    args.rgba_folder = os.path.join(args.output, 'images')
    args.undistorted_folder = os.path.join(args.output, 'undistorted')
    args.sparse_folder = os.path.join(args.output, 'sparse')
    args.database_path = os.path.join(args.output, args.colmap_db)
    csv_path = os.path.join(args.output, "camera.csv")

    if not os.path.exists(args.mask_folder):
        if "mask" in data["bands"]:
            url = os.path.join( input_folder, data["bands"]["mask"]["url"] )
            create_dir(args.mask_folder)
            extract_frames_from_video(url, args.mask_folder, extension="png", invert=True)

    if not os.path.exists(args.rgba_folder):
        if "rgba" in data["bands"]:
            url = os.path.join( input_folder, data["bands"]["rgba"]["url"] )
            create_dir(args.rgba_folder)
            extract_frames_from_video(url, args.rgba_folder)
        else:
            extract_frames(args)
        
    if os.path.exists(args.sparse_folder):
        if args.force:
            shutil.rmtree(args.sparse_folder)
    
    run_colmap(args)

    if not os.path.exists(csv_path):
        sparsedir = os.path.join(args.output, "sparse")
        N = len(os.listdir(args.rgba_folder))
        convert_to_csv(sparsedir, csv_path, expected_N=N)