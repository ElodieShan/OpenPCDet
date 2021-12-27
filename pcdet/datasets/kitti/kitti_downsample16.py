import copy
import pickle
from pathlib import Path

import numpy as np
from pcdet.utils import pointcloud_utils, pointcloud_sample_utils
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

def init_data_dir(path):
    import os
    file_root_path, file_name=os.path.split(path)
    if not os.path.exists(file_root_path):
        try:
            os.makedirs(file_root_path)
        except:
            print("Error occurs when make dirs %s"%file_root_path)

def create_gt_database(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_16lines_infos_%s.pkl' % train_split)
    print('---------------Start to generate data infos---------------')

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'kitti_downsample_16lines':
        import yaml
        from tqdm import tqdm
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2]),Loader=yaml.FullLoader)) # elodie
        
        data_root = dataset_cfg.DATA_PATH
        sample_data_dir = data_root + "/training/velodyne_16lines/"
        init_data_dir(sample_data_dir)
        for mode, info_path_ in dataset_cfg.INFO_PATH.items():

            # info_path = data_root +'/'+ info_path_[0]
            # with open(info_path, 'rb') as f:
            #     infos = pickle.load(f)
            # for i in tqdm(range(len(infos))):
            #     info = infos[i]
            #     if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
            #         info['point_cloud']['num_features'] += 1
            #         points_path = data_root + "/training/velodyne_ring/" + str(info['point_cloud']['lidar_idx']) + ".bin"
            #         points = np.fromfile(points_path,dtype=np.float32).reshape([-1, info['point_cloud']['num_features']])
            #         points = pointcloud_sample_utils.downsample_kitti(points, points[:,-1], verticle_switch=True, horizontal_switch=True)
            #         sample_data_path = sample_data_dir + str(info['point_cloud']['lidar_idx']) + ".bin"
            #         points.astype(np.float32).tofile(sample_data_path)
            # sample_info_path = data_root +'/'+ info_path_[0].replace('kitti','kitti_16lines')
            # with open(sample_info_path, "wb") as f:
            #     pickle.dump(infos, f)
            # print("-----------------Successfully dump ring info in path ",sample_info_path)

            # 先创建info和pcd数据，完成后再软链接到相应的downsample数据目录后运行下面的代码
            # 再把cfg的yaml文件中的DATA_PATH改成下采样后的数据目录
            if mode == "train":
                create_gt_database(
                    dataset_cfg=dataset_cfg,
                    class_names=['Car', 'Pedestrian', 'Cyclist'],
                    data_path=Path(data_root),
                    save_path=Path(data_root)
                )