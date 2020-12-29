import copy
import pickle
from pathlib import Path
import numpy as np
from pcdet.utils import pointcloud_utils 
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

def init_data_dir(path):
    import os
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def add_range_feature_kitti(points):
    if points.shape[1] == 4: # add feature angle/distance/lines
        horizontal_angles = pointcloud_utils.get_horizontal_angle(points[:,0],points[:,1])
        distances_3d = pointcloud_utils.get_distances_3d(points)
        vertical_angles = pointcloud_utils.get_vertical_angle(points[:,2], pointcloud_utils.get_distances_2d(points))
        ring = 1
        ring_list = [ring]
        for i in range(1,points.shape[0]):
            if horizontal_angles[i-1]<0 and horizontal_angles[i]>0:
            # if (horizontal_angles[i]>0 and (horizontal_angles[i-1] - horizontal_angles[i] > 10)) or (horizontal_angles[i-1]<0 and horizontal_angles[i]>0):
                ring += 1
            ring_list.append(ring)
        points = np.hstack((points, horizontal_angles.reshape(-1,1), distances_3d.reshape(-1,1), vertical_angles.reshape(-1,1), np.array(ring_list).reshape(-1,1)))
    return points


def create_gt_database(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_range_infos_%s.pkl' % train_split)
    print('---------------Start to generate data infos---------------')

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'kitti_add_range':
        import yaml
        from tqdm import tqdm
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2]),Loader=yaml.FullLoader)) # elodie
        
        data_root = dataset_cfg.DATA_PATH
        range_data_dir = data_root + "/training/velodyne_range/"
        init_data_dir(range_data_dir)
        for mode, info_path_ in dataset_cfg.INFO_PATH.items():

            # info_path = data_root +'/'+ info_path_[0]
            # with open(info_path, 'rb') as f:
            #     infos = pickle.load(f)
            # for i in tqdm(range(len(infos))):
            #     info = infos[i]
            #     if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
            #         points_path = data_root + "/training/velodyne/" + str(info['point_cloud']['lidar_idx']) + ".bin"
            #         points = np.fromfile(points_path,dtype=np.float32).reshape([-1, 4])
            #         points = add_range_feature_kitti(points)
            #         info['point_cloud']['num_features'] += 4
            #         range_data_path = range_data_dir + str(info['point_cloud']['lidar_idx']) + ".bin"
            #         points.astype(np.float32).tofile(range_data_path)
            # range_info_path = data_root +'/'+ info_path_[0].replace('kitti','kitti_range')
            # with open(range_info_path, "wb") as f:
            #     pickle.dump(infos, f)
            # print("-----------------Successfully dump range info in path ",range_info_path)

            # 先创建info和pcd数据，完成后再软链接到相应的range数据目录后运行下面的代码
            # 再把cfg的yaml文件中的DATA_PATH改成下采样后的数据目录
            if mode == "train":
                create_gt_database(
                    dataset_cfg=dataset_cfg,
                    class_names=['Car', 'Pedestrian', 'Cyclist'],
                    data_path=Path(data_root),
                    save_path=Path(data_root)
                )