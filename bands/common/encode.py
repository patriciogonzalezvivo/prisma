# Copyright (c) 2024, Patricio Gonzalez Vivo
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (the "License"). 
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#


import numpy as np
import math
import cv2


def hue_to_rgb(hue):
    """Convert hue to RGB."""
    if isinstance(hue, np.ndarray):
        rgb = np.zeros((hue.shape[0], hue.shape[1], 3))
        rgb[..., 0] = hue * 6.0
        rgb[..., 1] = hue * 6.0 + 4.0
        rgb[..., 2] = hue * 6.0 + 2.0
    else:
        rgb = np.zeros(3)
        rgb[0] = hue * 6.0
        rgb[1] = hue * 6.0 + 4.0
        rgb[2] = hue * 6.0 + 2.0

    rgb = np.abs(np.mod(rgb, 6.0) - 3.0) - 1.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def heat_to_rgb(heat):
    """Convert heat to RGB."""
    return hue_to_rgb( (1.0-heat) * 0.65 )


def rgb_to_hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def rgb_to_heat(rgb):
    """Convert RGB to heat."""
    hue = rgb_to_hsv(rgb)[...,0] / 360.0
    return np.clip(1.0 - hue * 1.538461538, 0.0, 1.0);


def mask_to_rgb(m: np.ndarray ):
    """Convert mask to RGB."""
    masks = np.where(m == 1, 255, m)
    return np.stack([masks] * 3, axis=-1)


def saturation(rgb, sat):
    """Set saturation."""
    rgb[..., 0] = rgb[..., 0] * sat + (1.0-sat)
    rgb[..., 1] = rgb[..., 1] * sat + (1.0-sat)
    rgb[..., 2] = rgb[..., 2] * sat + (1.0-sat)
    return rgb


def float_to_edge(channel, ksize=1):

    img = (channel * 255).astype(np.uint8)

    # Calculation of Sobelx 
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=ksize)

    # Calculation of Sobely 
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=ksize)

    sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    
    sobel_mag *= 255.0 / sobel_mag.max()

    return sobel_mag / 255.0


def encode_polar(a: np.ndarray, rad):
    """Encode polar cordinates to Hue (angle) and Saturation (radius)."""
    rgb = hue_to_rgb(a)
    rgb = saturation(rgb, rad)
    return rgb


def encode_flow(flow, mask):
    """Encode flow to RGB."""
    flow = 2**15 + flow * (2**8)
    mask &= np.max(flow, axis=-1) < (2**16 - 1)
    mask &= 0 < np.min(flow, axis=-1)
    return np.concatenate([flow.astype(np.uint16), mask[..., None].astype(np.uint16) * (2**16 - 1)], axis=-1)


def process_flow(flow):
    """Process flow."""

    h, w, _ = flow.shape
    distances = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))

    max_distance = distances.max()
    dX = flow[..., 0] / float(max_distance)
    dY = flow[..., 1] / float(max_distance)
    rad = np.sqrt(np.square(dX) + np.square(dY))
    a = (np.arctan2(dY, dX) / np.pi + 1.0) * 0.5
    rgb = encode_polar(a, rad)

    return (rgb * 255).astype(np.uint8), max_distance


def nearestPowerOfTwo(x):
    """Get nearest power of two."""
    return int(math.pow(2, math.ceil( math.log(x) / math.log(2) )))


def get_uv_form_index(index, img_size):
    """Get UV from index."""
    x = index % img_size
    y = math.floor(index / img_size)
    return x, y


def float_to_rgb(value, min_value=0.0, max_value=1.0, base=256):
    """Convert float to RGB."""
    L  = np.clip( (value - min_value) / (max_value - min_value), 0.0, 1.0) * (base * base * base - 1)
    return (    (np.floor(L % base)) / (base - 1),
                (np.floor(L / base) % base) / (base - 1),
                (np.floor(L / (base * base)) % base) / (base - 1) )


def encode_data_into_img(data, min_value=0.0, max_value=1.0, base=256, gain=1.0):
    """Encode data into image."""
    
    if isinstance(data, list):
        data = np.array(data)

    total_elements = data.shape[0]

    # if data have only one dimension
    if len(data.shape) == 1:
        variables_per_element = 1
    else:
        variables_per_element = data.shape[1]

    print("total_elements:", total_elements)
    print("variables_per_element:", variables_per_element)
    print("min_value:", np.min(data, axis=0))
    print("max_value:", np.max(data, axis=0))

    if variables_per_element > 1:
        if isinstance(min_value, float):
            # create a list of min_value of the same size as elements in data
            min_value = np.full(variables_per_element, min_value)

        elif isinstance(min_value, list):
            min_value = np.array(min_value)

        if isinstance(max_value, float):
            # create a list of max_value of the same size as elements in data
            max_value = np.full(variables_per_element, max_value)

        elif isinstance(max_value, list):
            max_value = np.array(max_value)

    print("min_value:", min_value)
    print("max_value:", max_value)
    
    img_size = int( nearestPowerOfTwo( math.sqrt( total_elements ) ) )
    print("img_size:", img_size)
    img = np.zeros((img_size, img_size, max(3, variables_per_element)))
    i = 0

    if variables_per_element == 1:
        pack_resolution = base * base * base
        for value in data:
            
            if gain != 1.0:
                value   = value * gain;
            
        
            x, y = get_uv_form_index(i, img_size)
            i += 1

            # L  = np.clip(value / max_value, 0.0, 1.0) * pack_resolution
            # img[y,x] = (    (np.floor(L % base)) / (base - 1),
            #                 (np.floor(L / base) % base) / (base - 1),
            #                 (np.floor(L / (base * base)) % base) / (base - 1) )
            img[x,y] = float_to_rgb(value, 0, max_value, base)
            
    elif variables_per_element == 3:
        delta = max_value - min_value

        for value in data:
            x, y = get_uv_form_index(i, img_size)
            i += 1

            img[y,x] = (    (value[0] - min_value[0]) / delta[0],
                            (value[1] - min_value[1]) / delta[1],
                            (value[2] - min_value[2]) / delta[2] )

    elif variables_per_element == 4:
        delta = max_value - min_value

        for value in data:
            x, y = get_uv_form_index(i, img_size)
            i += 1

            img[y,x] = (    (value[0] - min_value[0]) / delta[0],
                            (value[1] - min_value[1]) / delta[1],
                            (value[2] - min_value[2]) / delta[2],
                            (value[3] - min_value[3]) / delta[3] )

    return img