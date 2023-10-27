import math
import random
import hashlib
import logging
from enum import Enum

import cv2
import numpy as np

from utils.data_utils import LinearRamp
from metrics.evaluation.masks.mask import SegmentationMask

LOGGER = logging.getLogger(__name__)


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(shape, max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(shape, margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)


def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


class RandomSuperresMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, shape, iter_i=None):
        return make_random_superres_mask(shape, **self.kwargs)


class MixedMaskGenerator:
    def __init__(self, irregular_proba=1/3, hole_range=[0,0,0.7], irregular_kwargs=None,
                 box_proba=1/3, box_kwargs=None,
                 segm_proba=1/3, segm_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []
        self.hole_range = hole_range

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, shape, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(shape, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        if np.mean(result) <= self.hole_range[0] or np.mean(result) >= self.hole_range[1]:
            return self.__call__(shape, iter_i=iter_i, raw_image=raw_image)
        else:
            return result


class RandomSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):
        
        masks = self.impl.get_masks(img)
        fil_masks = []
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            fil_masks.append(cur_mask)

        mask_index = np.random.choice(len(fil_masks),
                                        size=1,
                                        replace=False)
        mask = fil_masks[mask_index]

        return mask


class SegMaskGenerator:
    def __init__(self, hole_range=[0.1, 0.2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = RandomSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range

    def __call__(self, img, iter_i=None, raw_image=None):
        result = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        return result

class FGSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):
        
        masks = self.impl.get_masks(img)
        mask = masks[0]
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            mask += cur_mask
        
        mask = mask > 0
        return mask

class SegBGMaskGenerator:
    def __init__(self, hole_range=[0.1, 0.2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = FGSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range
        self.cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 1.0],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 250,
                'max_width': 150,
                'max_times': 3,
                'min_times': 1,
            },
            'box_proba': 0,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**self.cfg)

    def __call__(self, img, iter_i=None, raw_image=None):
        shape = img.shape[:2]
        mask_fg = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        bg_ratio = 1 - np.mean(mask_fg)
        result = self.bg_mask_gen(shape, iter_i=iter_i, raw_image=raw_image)
        result = result - mask_fg
        if np.mean(result) <= self.hole_range[0]*bg_ratio or np.mean(result) >= self.hole_range[1]*bg_ratio:
            return self.__call__(shape, iter_i=iter_i, raw_image=raw_image)
        return result


def get_mask_generator(kind, cfg=None):
    if kind is None:
        kind = "mixed"
    
    if cfg is None:
        cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 0.7],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 200,
                'max_width': 100,
                'max_times': 5,
                'min_times': 1,
            },
            'box_proba': 1,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            },
            'segm_proba': 0,}

    if kind == "mixed":
        cl = MixedMaskGenerator
    elif kind =="segmentation":
        cl = SegBGMaskGenerator
    else:
        raise NotImplementedError(f"No such generator kind = {kind}")
    return cl(**cfg)