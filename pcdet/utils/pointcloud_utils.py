import numpy as np

""" Function: get_distance_2d
@breif 单点根据x、y计算2D距离
"""
def get_distance_2d(x, y):
    return np.sqrt(np.sum([x**2,y**2]))

def get_distances_2d(points):
    return np.sqrt(np.sum([points[:,0]**2,points[:,1]**2],axis=0))

""" Function: get_distance_3d
@breif 单点根据x、y、z计算3D距离
"""
def get_distance_3d(x, y, z):
    return np.sqrt(np.sum([x**2,y**2,z**2]))

""" Function: get_horizontal_angle
@breif 单点/多点根据x、y计算水平角度，
@param x、y可以是一个值，也可以是np.array
"""
def get_horizontal_angle(x, y):
    return np.arctan2(y,x)/np.pi*180

def get_horizontal_angles(points):
    return np.arctan2(points[:,1],points[:,0])/np.pi*180
""" Function: get_vertical_angle
@breif 单点/多点根据z、distance2d计算垂直角度，
@param z、distance2d可以是一个值，也可以是np.array
"""
def get_vertical_angle(z, distance2d):
    return np.arctan2(z,distance2d)/np.pi*180

""" Function: get_distances_3d
@breif 多点根据x、y计算水平距离，
@param x、y输入类型是np.array
"""
def get_distances_3d(points):
    return np.sqrt(np.sum([points[:,0]**2,points[:,1]**2,points[:,2]**2],axis=0))

""" Function: get_pulsewidth
@breif 多点提取脉宽信息
"""
def get_pulsewidth(points):
    return points[:,3]

""" Function: get_layer
@breif 多点提取通道号信息
"""
def get_layer(points):
    return points[:,5]

"""
@breif 滤除距离和脉宽大于某阈值的点

@param　distance3d:　距离
@param　pulsewidth:　脉宽
@param　distance_filter:　距离阈值
@param　pulsewidth_filter:　脉宽阈值
"""
def filter_invalid_points(distance3d, pulsewidth, distance_filter_min, distance_filter_max, pulsewidth_filter):
    invalid_points_mask = np.vstack((np.array(distance3d < distance_filter_max), np.array(distance3d>distance_filter_min), np.array(pulsewidth < pulsewidth_filter), np.array(pulsewidth > 0)))
    invalid_points_mask = invalid_points_mask.all(axis=0)
    return invalid_points_mask

"""
@breif 获取某个角度上的点

@param　horizontal_angles:　水平角度
@param　angle_analyze:　筛选的角度值
"""
def filter_points_by_angle(horizontal_angles, angle_analyze):
    angle_mask = np.vstack((np.array(horizontal_angles < (angle_analyze + 0.07 )), np.array(horizontal_angles > (angle_analyze - 0.07 ))))
    angle_mask = angle_mask.all(axis=0)
    return angle_mask

"""
@breif 将数据从n*1转换为m*16，按照通道号存储
"""
def reform_data_by_layer(data, layer):
    return [data[layer==i+1] for i in range(16)]

"""
@breif 计算距离值和脉宽值的测距均值

@param　distance3d:　距离
@param　pulsewidth:　脉宽
@param　layer:　通道号
"""
def get_mean_by_layer(distance3d, pulsewidth, layer):
    distance_mean_byseq = np.array([np.mean(distance3d[layer==i+1]) for i in range(16)])
    distance_mean_byseq[np.isnan(distance_mean_byseq)] = 0
    # -------------------------若平均值为nan，则赋值为0
    pulsewidth_mean_byseq = np.array([np.mean(pulsewidth[layer==i+1]) for i in range(16)])
    pulsewidth_mean_byseq[np.isnan(pulsewidth_mean_byseq)] = 0
    return distance_mean_byseq, pulsewidth_mean_byseq

