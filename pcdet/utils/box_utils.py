import numpy as np
import scipy
import torch
import copy
from scipy.spatial import Delaunay

from . import common_utils



def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask



def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)
