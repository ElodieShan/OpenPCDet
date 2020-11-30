import numpy as np
from .pointcloud_utils import *

def downsample_kitti(points, ring, verticle_switch=True, horizontal_switch=True, return_extra_points=False):
    extra_points = None
    if verticle_switch:
        ring_remained = [33, 32, 29, 27, 25, 23, 21, 19, 16, 14, 12, 10, 8, 6, 4, 2]
        mask = np.in1d(ring,ring_remained)
        if return_extra_points:
            extra_points = points[~mask]
        points = points[mask] # faster
        ring = ring[mask]
    if horizontal_switch:
        distances = np.array(get_distances_2d(points)).T
        distances = np.fabs(distances[1:] - distances[:-1])
        np.append(distances,0)
        half_mask = np.arange(0,points.shape[0]-1,2)
        mask = np.ones(points.shape[0], dtype=bool)
        mask[half_mask[np.any((distances[half_mask] < 0.1,distances[half_mask-1] < 0.1),axis=0)]] = False
        if return_extra_points:
            extra_points = points[~mask] if extra_points is None else np.vstack((extra_points,points[~mask]))
        points = points[mask]
    return points, extra_points

def downsample_kitti_v2(points, ring, verticle_switch=True, horizontal_switch=True):
    if verticle_switch:
        ring_remained = [6, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37]        
        # ring_remained = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 38]
        mask = np.in1d(ring,ring_remained)
        points = points[mask] # faster
        ring = ring[mask]
    if horizontal_switch:
        distances = np.array(get_distances_2d(points)).T
        distances = np.fabs(distances[1:] - distances[:-1])
        np.append(distances,0)
        half_mask = np.arange(0,points.shape[0]-1,2)
        mask = np.ones(points.shape[0], dtype=bool)
        mask[half_mask[np.any((distances[half_mask] < 0.1,distances[half_mask-1] < 0.1),axis=0)]] = False
        points = points[mask]
    return points

def downsample_kitti_to_VLP16(points, ring, verticle_switch=True):
    if verticle_switch:
        ring_remained = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]
        mask = np.in1d(ring,ring_remained)
        points = points[mask] # faster
        ring = ring[mask]
    return points

def downsample_nusc_v2(points, ring):
    # if points.shape[1]!=5:
    #     print("points attribution do not contains ring num..")
    #     return points
    points_mask = np.all([ring>16 ,ring<26],axis=0)
    points = points[points_mask]
    return points

def upsample_nusc_v1(points, ring):
    if points.shape[0]>0:
        distances = np.array(get_distances_2d(points)).T
        points_upsample = (points[:-1] + points[1:])/2
        mask = np.all((ring[:-1]!=25, abs(distances[1:] - distances[:-1])<0.1, distances[1:]>0.1 , distances[:-1]>0.1),axis=0)
        points_upsample = points_upsample[mask]
        if points_upsample.shape[0]>0:
            # points_upsample[:,4] = 0
            points = np.vstack((points, points_upsample))
    return points
