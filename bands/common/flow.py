
import os
import numpy as np

import cv2

import torch
import torch.nn.functional as F

from common.encode import process_flow, encode_flow


def load_image(image):
    img = np.array(image)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow, alpha_1=0.05, alpha_2=0.5):
    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    

def write_flow(args, 
                fwd_flow, fwd_flow_video, max_disps, idx,
                fwd_mask=None, fwd_mask_video=None,
                bwd_flow=None, bwd_flow_video=None, 
                bwd_mask=None, bwd_mask_video=None):
    flow_pixels, max_disp = process_flow(fwd_flow)

    fwd_flow_video.write(flow_pixels)
    max_disps.append(max_disp)

    if fwd_mask is not None and fwd_mask_video:
        # fwd_mask_video.write(encode_flow(fwd_flow, fwd_mask))
        mask = np.where(fwd_mask == 1, 255, fwd_mask)
        mask_rgb = np.stack([mask] * 3, axis=-1)
        fwd_mask_video.write(mask_rgb)

    if bwd_mask is not None and bwd_mask_video:
        # bwd_mask_video.write(encode_flow(bwd_flow, bwd_mask))
        mask = np.where(bwd_mask == 1, 255, bwd_mask)
        mask_rgb = np.stack([mask] * 3, axis=-1)
        bwd_mask_video.write(mask_rgb)
        
    if args.backwards and bwd_flow_video:
        bwd_flow_pixels, _ = process_flow(bwd_flow)
        bwd_flow_video.write(bwd_flow_pixels)

    if args.subpath != '':
        write_flow(os.path.join(args.subpath + '_fwd', '%04d.flo' % idx), fwd_flow)
        if args.backwards:
            write_flow(os.path.join(args.subpath + '_bwd', '%04d.flo' % idx), bwd_flow)

    if args.subpath_mask != '':
        cv2.imwrite(os.path.join(args.subpath_mask + '_fwd', '%04d.png' % idx), encode_flow(fwd_flow, fwd_mask))
        if args.backwards:
            cv2.imwrite(os.path.join(args.subpath_mask + '_bwd', '%04d.png' % idx), encode_flow(bwd_flow, bwd_mask))

    # if args.subpath_vis != '':
    #     cv2.imwrite(os.path.join(args.subpath_vis + '_fwd', '%04d.jpg' % idx), flow_to_image(fwd_flow))
    #     if args.backwards:
    #         cv2.imwrite(os.path.join(args.subpath_vis + '_bwd', '%04d.jpg' % idx), flow_to_image(bwd_flow))