import matplotlib.pyplot as plt
import cv2
import numpy as np
from metric3d.utils.transform import gray_to_colormap
import torch

def save_raw_imgs( 
    pred: torch.tensor,  
    rgb: torch.tensor, 
    filename: str,
    scale: float=200.0, 
    ):
    """
    Save raw GT, predictions, RGB in the same file.
    """
    cv2.imwrite(filename + '_rgb.jpg', rgb)
    cv2.imwrite(filename + '_d.png', (pred*scale).astype(np.uint16))
    

def save_val_imgs(
    pred: torch.tensor, 
    rgb: torch.tensor, 
    filename: str,
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, pred_color = get_data_for_log(pred, rgb)
    rgb = rgb.transpose((1, 2, 0))
    plt.imsave(filename + '_depth.jpg', pred_color)
    cat_img = np.concatenate([rgb, pred_color], axis=0)
    plt.imsave(filename + '_merge.jpg', cat_img)


def get_data_for_log(pred: torch.tensor, rgb: torch.tensor):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std = np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()

    pred[pred<0] = 0
    max_scale = pred.max()
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, pred_color
