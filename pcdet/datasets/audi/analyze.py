import copy
import pickle
from pathlib import Path
import numpy as np
from pcdet.utils import pointcloud_utils 
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

if __name__ == '__main__':
    train_info_path = "/home/elodie/OpenPCDet/data/audi/audi_infos_train.pkl"
    # train_info_path = "/home/elodie/OpenPCDet/data/kitti/kitti_infos_train.pkl"
    with open(train_info_path, 'rb') as f:
        train_infos = pickle.load(f)

    class_type_list = ['Car', 'Pedestrian', 'Cyclist']
    class_bbox = []
    for class_type in class_type_list:
        class_bbox = []
        for i in range(len(train_infos)):
            annos = train_infos[i]['annos']
            gt_name = annos['name']
            gt_box = annos['gt_boxes_lidar']
            mask = annos['name'][annos['name']!='DontCare']==class_type
            
            if gt_box.shape[0]>0 and gt_box[mask].shape[0]>0:
                for box in gt_box[mask]:
                    class_bbox.append(box)

        class_bbox = np.array(class_bbox).reshape(-1,7)
        print('----', class_type,'--------')
        print(class_bbox.shape)
        print("avg size:",np.average(class_bbox,axis=0))


# if __name__ == '__main__':
#     train_info_path = "/home/elodie/OpenPCDet/output/audi_models/second_16lines_audi/21030401-Audi-80epoch-batch4/eval/epoch_26/val/test/result_w_iou.pkl"
#     # train_info_path = "/home/elodie/OpenPCDet/data/kitti/kitti_infos_train.pkl"
#     with open(train_info_path, 'rb') as f:
#         train_infos = pickle.load(f)

#     class_type_list = ['Car']

#     # class_type_list = ['Car', 'Pedestrian', 'Cyclist']
#     class_bbox = []
#     for class_type in class_type_list:
#         class_bbox = []
#         class_bbox_iou7 = []
#         for i in range(len(train_infos)):
#             annos = train_infos[i]
#             gt_name = annos['name']
#             gt_box = annos['boxes_lidar']
#             mask = annos['name'][annos['name']!='DontCare']==class_type
#             iou = annos['iou']
#             print()
#             mask2 = annos['iou'][mask]>=0.7
#             if gt_box.shape[0]>0 and gt_box[mask].shape[0]>0:
#                 for box in gt_box[mask]:
#                     class_bbox.append(box)
#             if gt_box[mask][mask2].shape[0]>0:
#                 for box in gt_box[mask][mask2]:
#                     class_bbox_iou7.append(box)
#         class_bbox = np.array(class_bbox).reshape(-1,7)
#         class_bbox_iou7 = np.array(class_bbox_iou7).reshape(-1,7)

#         print('----', class_type,'--------')
#         print(class_bbox.shape)
#         print(class_bbox_iou7.shape)

#         print("avg size:",np.average(class_bbox,axis=0))

"""
Audi
---- Car --------
(14383, 7)
avg size: [20.45286845  0.20702032 -0.87078641  3.96402141  1.99950914  1.7031419
  0.76265772]
---- Pedestrian --------
(2502, 7)
avg size: [22.27102271 -1.17850427 -0.80981495  0.97753397  0.7759992   1.81329736
  1.37335581]
---- Cyclist --------
(184, 7)
avg size: [28.21080562 -0.2376763  -0.84928347  1.95396739  0.86657609  1.82266304
  1.53002683]

---- Car --------
(14383, 7)
avg size: [20.45286845  0.20702032 -0.87078641  3.96402141  1.99950914  1.7031419
  0.17082193]
---- Pedestrian --------
(2502, 7)
avg size: [22.27102271 -1.17850427 -0.80981495  0.97753397  0.7759992   1.81329736
  0.12404986]
---- Cyclist --------
(184, 7)
avg size: [28.21080562 -0.2376763  -0.84928347  1.95396739  0.86657609  1.82266304
  0.18960558]

KITTI

---- Car --------
(14357, 7)
avg size: [29.63470723  2.43797646 -0.73492998  3.89206519  1.61876715  1.52986348
 -1.59344939]
---- Pedestrian --------
(2207, 7)
avg size: [19.06900047  0.05188316 -0.57548763  0.81796103  0.62783416  1.76773901
 -1.83260421]
---- Cyclist --------
(734, 7)
avg size: [21.78718119 -2.84906466 -0.66564166  1.77081744  0.56961853  1.72332425
 -1.04623229]
"""