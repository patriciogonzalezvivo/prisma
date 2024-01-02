
![prisma](https://github.com/patriciogonzalezvivo/prisma/assets/346914/e268e0c5-ce0e-4988-bda8-fcd136a7876f)



# PRISMA

Small framework to infer different types of data (aka: "bands") from a single image/video. 
It's a combination of open sourced models to infer:

    * depth (MiDAS v3.1, ZoeDepth, Marigold, PatchFusion)
    * optical flow (RAFT)
    * segmentation (mmdet, detectron2, mediapipe)
    * inpainting (FcfGAN)
    * super resolution (RealSrGAN)

Each image or video will become a folder where all the derived bands will be stored together with a `meta.json` that keep tracks of the associated data.

```Shell
conda env create -f environment.yml
conda activate prisma
sh download_models.sh

# Install mmcv (for mmdetection)
pip install -U openmim
mim install mmengine
mim install "mmcv-full==1.7.1"
```

## Depth estimation (MiDAS 3.1)

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


## Depth Estimation (ZoeDepth)

**Paper:** [Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288)

**License:** [MIT](bands/patchfusion/zoedepth/LICENSE)

**Code Repo:** [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)

**Use:**

```Shell
depth_zoe.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
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

## Depth Estimation (PatchFusion)

**Paper:** [PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation](https://zhyever.github.io/patchfusion/images/paper.pdf)

**License:** [MIT](bands/patchfusion/LICENSE)

**Code Repo:** [zhyever/PatchFusion](https://github.com/zhyever/PatchFusion)

**Use:**

```Shell
depth_fusion.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
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

## Depth Estimation (Marigold)

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


## Optical Flow (RAFT)

Based on https://github.com/SharifElfouly/opical-flow-estimation-with-RAFT

Seems to be very good: [Optical Flow Estimation Benchmark](https://paperswithcode.com/sota/optical-flow-estimation-on-sintel-clean)

**Paper:** [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039)

**License:** [BSD](bands/raft/LICENSE)

**Code Repo:** [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

**Use:**

```Shell
flow.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

## Segmentation (MMDetection)

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


## FcF-Inpainting

**Paper:** [eys to Better Image Inpainting: Structure and Texture Go Hand in Hand](https://praeclarumjj3.github.io/fcf-inpainting/)

**Code Repo:** [SHI-Labs/FcF-Inpainting](https://github.com/SHI-Labs/FcF-Inpainting)

**License:** [Apaches](https://github.com/SHI-Labs/FcF-Inpainting/blob/main/LICENSE)

**Use:**

```Shell
inpaint_fcfgan.py -input <IMAGE/VIDEO> -mask <IMAGE/VIDEO> -output <IMAGE/VIDEO> 
```

Citation:

```
@inproceedings{jain2022keys,
    title={Keys to Better Image Inpainting: Structure and Texture Go Hand in Hand},
    author={Jitesh Jain and Yuqian Zhou and Ning Yu and Humphrey Shi},
    booktitle={WACV},
    year={2023}
} 
```


## Super Resolution

**Code Repo:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

**License:** [BSD](bands/realesrgan/LICENSE)

Citation:
```
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

