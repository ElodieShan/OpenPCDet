import copy
import pickle
from pathlib import Path
import numpy as np
from pcdet.utils import pointcloud_utils 
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

if __name__ == '__main__':
    train_info_path = "/home/elodie/OpenPCDet/data/audi_mimic/audi_infos_train.pkl"
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
#     train_info_path = "/home/elodie/OpenPCDet/output/audi_models/second_audi/21030507-Audi-80epoch-batch4/eval/epoch_80/val/test_trainset_vansuv_to_car/result_w_iou.pkl"
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
#             mask2 = annos['iou'][mask]>=0.5
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
(12327, 7)
avg size: [20.70480574  0.48652676 -0.87196455  3.92929423  1.99493875  1.68559909
  0.16928482]
---- Pedestrian --------
(1871, 7)
avg size: [22.89038213 -1.2977962  -0.78718548  0.68825762  0.66760021  1.82228755
  0.15282161]
---- Cyclist --------
(1023, 7)
avg size: [21.71216881 -2.14966805 -0.91618603  1.70584555  0.72492669  1.62160313
  0.3673672 ]


--- AUDI with van

---- Car --------
(13295, 7)
avg size: [20.90520264  0.49922966 -0.85977159  3.95355322  2.00530199  1.70738398
  0.17182453]
---- Pedestrian --------
(1871, 7)
avg size: [22.89038213 -1.2977962  -0.78718548  0.68825762  0.66760021  1.82228755
  0.15282161]
---- Cyclist --------
(1023, 7)
avg size: [21.71216881 -2.14966805 -0.91618603  1.70584555  0.72492669  1.62160313
  0.3673672 ]

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