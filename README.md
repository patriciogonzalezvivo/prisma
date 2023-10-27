# PRISMA

Series of scripts to derive different computer vision tasks from a single image or video.

```Shell
python -m pip install -r requirements.txt
sh download_models.sh
```


## Optical Flow (RAFT)

Based on https://github.com/SharifElfouly/opical-flow-estimation-with-RAFT

Seems to be very good: [Optical Flow Estimation Benchmark](https://paperswithcode.com/sota/optical-flow-estimation-on-sintel-clean)

**Paper:** [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039)

**Code Repo:** [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)

**Use:**

```Shell
flow.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```


## Depth estimation (MiDAS 3.1)

**Paper:** [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341v3)

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


## Normals

**Code Repo:** [Surface Normal Uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty)

**Use:**

```Shell
normal.py --input <IMAGE/VIDEO> --output <IMAGE/VIDEO>
```

Citation:

```
@InProceedings{Bae2021,
    title   = {Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation}
    author  = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2021}                         
}
```


## Segmentation 

**Code Repo:** [MMDetection](https://github.com/open-mmlab/mmdetection)

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

Citation for SOLOv2 model
```
@article{wang2020solov2,
  title={SOLOv2: Dynamic and Fast Instance Segmentation},
  author={Wang, Xinlong and Zhang, Rufeng and  Kong, Tao and Li, Lei and Shen, Chunhua},
  journal={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```


## FcF-Inpainting

**Paper:** [eys to Better Image Inpainting: Structure and Texture Go Hand in Hand](https://praeclarumjj3.github.io/fcf-inpainting/)

**Code Repo:** [SHI-Labs/FcF-Inpainting](https://github.com/SHI-Labs/FcF-Inpainting)

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

Citation:
```
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

