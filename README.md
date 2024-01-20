![_prisma_](https://github.com/patriciogonzalezvivo/prisma/assets/346914/0a468415-5a19-4993-a9ff-e8ee867fc629)

# PRISMA

PRISMA it's a pipeline for performing multiple inferences or computations (refere as "bands") from any image or video. 

It's a combination of different algorithms and open sourced pre-train models such as:

* depth (MiDAS v3.1, ZoeDepth, Marigold, PatchFusion)
* optical flow (RAFT)
* segmentation (mmdet)
* camera pose (colmap)

The resulting bands are stored in a folder with the same name as the input file. Each band is stored as a single `.png` or `.mp4` file. In the case of videos, at the last step will attempt to perform a sparse reconstruction which will can be use for both NeRFs (like [NVidia's Instant-ngp](https://github.com/NVlabs/instant-ngp)) or [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) training. 

Infered depth is exported by default as a heatmap that can be decoded realtime using [LYGIA's heatmap GLSL/HLSL sampling](https://lygia.xyz/sample/heatmap). While the optical flow is encoded as HUE (angle) and saturation which also can be decoded realtime using [LYGIA opticalFlow GLSL/HLSL sampler](https://lygia.xyz/sample/opticalFlow).

## Install

Main dependencies:

* [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
* [COLMAP](https://colmap.github.io/install.html)

```Shell
git clone git@github.com:patriciogonzalezvivo/prisma.git
cd prisma

conda env create -f environment.yml
conda activate prisma
sh download_models.sh

# Install mmcv (for mmdetection)
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.1"
```

## How it works?

### a. Process

We start by processing an image or video. Let's start by processing an image:

```bash
python process.py -i data/gog.jpg
```

With out providing an `--output` this will create a folder with the same filename which will contain all the derived bands (`rgba`, `flow`, `mask` and `depth_*`).

```
gog.jpg
gog/
├── depth_patchfusion.png
├── mask.png
├── metadata.json
└── rgba.png
```

In the forlder you will find a `metadata.json` file that contains all the metadata associated with the original image or video.

```json
{
    "bands": {
        "rgba": {
            "url": "rgba.png"
        },
        "depth_patchfusion": {
            "url": "depth_patchfusion.png",
            "values": {
                "min": {
                    "value": 1.6147574186325073,
                    "type": "float"
                },
                "max": {
                    "value": 11.678544044494629,
                    "type": "float"
                }
            }
        },
        "mask": {
            "url": "mask.png",
            "ids": [
                "person",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe"
            ]
        }
    },
    "width": 934,
    "height": 440,
    "principal_point": [
        467.0,
        220.0
    ],
    "focal_length": 641.0616195031489,
    "field_of_view": 37.88246641919117
}
```

Currently PRISMA supports multiple depth estimation algorithms. You can select which one to use by providing the `--depth`|`-d` argument: `depth_midas`, `depth_zoedepth`, `depth_patchfusion`, `depth_marigold` or `all`. By defualt images will be processed using `depth_patchfusion`, while videos will use `depth_zoedepth`.

When processing videos, by default PRISMA creates the least ammount of data by creating a single `.png` or `.mp4` for each band. In the case of videos data like min/max values will be stored on `.cvs`.

it's possible to save extra data by setting the `--extra`|`-e` level number.

0. store bands as a single `.png` and `.mp4` (video have usually an associated `.csv` file)
1. store images as `.ply` point clouds, for videos it extracts the reslting frames as `.png`
2. store optical flow from videos as `.flo` files.
3. store inferenced depth as `.npy` files.

Let's try now extracting all depth models and individual frames from a video:

```bash
python process.py -i data/rocky.mp4 -d all -e 1
```

Which produce the folowing folder structure:

```
rocky.mp4
rocky/
├── depth_marigold/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── depth_marigold_max.csv
├── depth_marigold_min.csv
├── depth_marigold.mp4
├── depth_midas/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── depth_midas_max.csv
├── depth_midas_min.csv
├── depth_midas.mp4
├── depth_patchfusion/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── depth_patchfusion_max.csv
├── depth_patchfusion_min.csv
├── depth_patchfusion.mp4
├── depth_zoedepth/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── depth_zoedepth_max.csv
├── depth_zoedepth_min.csv
├── depth_zoedepth.mp4
├── flow/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── flow.csv
├── flow.mp4
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   ├── ...
│   └── 000110.png
├── mask/
|   ├── 000000.png
|   ├── 000001.png
|   ├── ...
|   └── 000110.png
├── mask.mp4
|── sparse/
|   └── 0/
|       ├── cameras.bin
|       ├── images.bin
|       ├── points3D.bin
|       └── points3D.txt
|── camera_pose.csv
|── colmap.db
├── metadata.json
└── rgba.mp4
```

### b. Visualize

View the resulting bands from the processed image/video using [ReRun](https://www.rerun.io/):

```bash

```bash
python view.py -i data/rocky
```

![2024-01-20 06-35-33](https://github.com/patriciogonzalezvivo/prisma/assets/346914/7abff827-631a-45cd-8aba-819172f59877)

### c. Concatenate bands

In order to export the bands as a single image or video you can use the `concat.py` script:

```bash
python concat.py -i data/gog -o test.png
```

![test](https://github.com/patriciogonzalezvivo/prisma/assets/346914/763d3ada-736c-4676-ad4f-55eafe9dcf40)


## Roadmap

[ ] Suport from Record3D RGB-D format

[ ] Find a proper way to extract Field of View, and centers from single images. Like EXIF in images or through a model.

## Licenses and Credits

This pipeline is Copyright (c) 2024, Patricio Gonzalez Vivo and Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) please reach out if you want to use it commercially.

All the models and software used by it are commercial ready licenses like MIT, Apache and BSD.

### Depth estimation (MiDAS 3.1)

**Paper:** [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341v3)

**License:** [MIT](bands/midas/LICENSE)

**Code Repo:** [isl-org/MiDaS](https://github.com/isl-org/MiDaS)

**Use:**

```Shell
depth_midas.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

Citation:
```
@ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```

Citation for DPT-based model:
```
@article{Ranftl2021,
    author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
    title     = {Vision Transformers for Dense Prediction},
    journal   = {ArXiv preprint},
    year      = {2021},
}
```


### Depth Estimation (ZoeDepth)

**Paper:** [Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)

**License:** [MIT](bands/patchfusion/zoedepth/LICENSE)

**Code Repo:** [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)

**Use:**

```Shell
depth_zoedepth.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

Citation
```
@misc{https://doi.org/10.48550/arxiv.2302.12288,
    doi = {10.48550/ARXIV.2302.12288},
    url = {https://arxiv.org/abs/2302.12288},
    author = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and Müller, Matthias},  
    keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},  
    publisher = {arXiv},
    year = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```

### Depth Estimation (PatchFusion)

**Paper:** [PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation](https://zhyever.github.io/patchfusion/images/paper.pdf)

**License:** [MIT](bands/patchfusion/LICENSE)

**Code Repo:** [zhyever/PatchFusion](https://github.com/zhyever/PatchFusion)

**Use:**

```Shell
depth_patchfusion.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

**Note:** [This pretrained model](https://huggingface.co/zhyever/PatchFusion/resolve/main/patchfusion_u4k.pt?download=true) needs to be downloaded and placed in the `models/` folder.


Citation

```
@article{li2023patchfusion,
    title={PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation}, 
    author={Zhenyu Li and Shariq Farooq Bhat and Peter Wonka},
    year={2023},
    eprint={2312.02284},
    archivePrefix={arXiv},
    primaryClass={cs.CV}}
```

### Depth Estimation (Marigold)

**Paper:** [Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://arxiv.org/abs/2312.02145)

**License:** [Apache](bands/marigold/LICENSE)

**Code Repo:** [prs-eth/Marigold](https://github.com/prs-eth/Marigold)

**Use:**

```Shell
depth_marigold.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

Citation

```bibtex
@misc{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation}, 
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      year={2023},
      eprint={2312.02145},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Optical Flow (RAFT)

Based on https://github.com/SharifElfouly/opical-flow-estimation-with-RAFT

Seems to be very good: [Optical Flow Estimation Benchmark](https://paperswithcode.com/sota/optical-flow-estimation-on-sintel-clean)

**Paper:** [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039)

**License:** [BSD](bands/raft/LICENSE)

**Code Repo:** [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

**Use:**

```Shell
flow.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

### Segmentation (MMDetection)

**Code Repo:** [MMDetection](https://github.com/open-mmlab/mmdetection)

**License:** [Apache](bands/mmdet/LICENSE)

**Use:**

```Shell
mask_mmdet.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

Citation:
```
@article{mmdetection,
    title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
    author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
              Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
              Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
              Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
              Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
              and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
    journal= {arXiv preprint arXiv:1906.07155},
    year={2019}
}
```
