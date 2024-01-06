import cv2
import numpy as np
from plyfile import PlyData, PlyElement

def create_point_cloud(depth, u0, v0, fx=1000.0, fy=1000.0):
    depth_blured = cv2.medianBlur(depth, 5)

    H, W = depth_blured.shape
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u = x - u0

    y_col = np.arange(0, H)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v = y - v0

    x = u / fx
    y = v / fy
    z = np.ones_like(x)
    pcl = np.stack([x, -y, -z], axis=2)

    return depth_blured[:, :, None] * pcl


def save_point_cloud(pcl, rgb, filename, binary=True):
    assert pcl.shape[0] == rgb.shape[0]

    points_3d = np.hstack((pcl, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # Format into Numpy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
        cur_point = points_3d[row_idx]
        vertices.append(
            tuple(
                dtype(point)
                for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # write
    PlyData([el], text=(not binary)).write(filename)
