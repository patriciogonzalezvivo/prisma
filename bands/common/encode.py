import numpy as np


def hue_to_rgb(hue):
    if isinstance(hue, np.ndarray):
        hue = 1.0 - hue
        rgb = np.zeros((hue.shape[0], hue.shape[1], 3))
        rgb[..., 0] = hue * 6.0
        rgb[..., 1] = hue * 6.0 + 4.0
        rgb[..., 2] = hue * 6.0 + 2.0
 
    else:
        hue = 1.0 - hue
        rgb = np.zeros(3)
        rgb[0] = hue * 6.0
        rgb[1] = hue * 6.0 + 4.0
        rgb[2] = hue * 6.0 + 2.0

    rgb = np.abs(np.mod(rgb, 6.0) - 3.0) - 1.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def heat_to_rgb(heat):
    return hue_to_rgb( 1.0 - heat * 0.65 )


def encode_polar(a: np.ndarray , rad):
    RGB = np.zeros((a.shape[0], a.shape[1], 3))
    RGB[..., 0] = a * 6.0
    RGB[..., 1] = a * 6.0 + 4.0
    RGB[..., 2] = a * 6.0 + 2.0

    RGB = np.abs( np.mod(RGB, 6.0) - 3.0 ) - 1.0
    RGB = np.clip(RGB, 0.0, 1.0)
    RGB[..., 0] = RGB[..., 0] * rad + (1.0-rad)
    RGB[..., 1] = RGB[..., 1] * rad + (1.0-rad)
    RGB[..., 2] = RGB[..., 2] * rad + (1.0-rad)
    return RGB


def encode_flow(flow, mask):
    flow = 2**15 + flow * (2**8)
    mask &= np.max(flow, axis=-1) < (2**16 - 1)
    mask &= 0 < np.min(flow, axis=-1)
    return np.concatenate([flow.astype(np.uint16), mask[..., None].astype(np.uint16) * (2**16 - 1)], axis=-1)


def process_flow(flow):
    h, w, _ = flow.shape
    distances = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))

    max_distance = distances.max()
    dX = flow[..., 0] / float(max_distance)
    dY = flow[..., 1] / float(max_distance)
    rad = np.sqrt(np.square(dX) + np.square(dY))
    a = (np.arctan2(dY, dX) / np.pi + 1.0) * 0.5
    rgb = encode_polar(a, rad)

    return (rgb * 255).astype(np.uint8), max_distance
