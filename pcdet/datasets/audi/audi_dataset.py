import copy
import pickle
from pathlib import Path

import numpy as np
from skimage import io

import torch
import glob
from os.path import join
import json

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class AudiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        # split_dir = Path.cwd() / 'split' / (self.split + '.txt')
        split_dir = Path(__file__).resolve().parent / 'split' / (self.split + '.txt')
        self.sample_scene_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.audi_infos = []
        self.include_audi_data(self.mode)

    def include_audi_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Audi dataset')
        audi_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                audi_infos.extend(infos)

        self.audi_infos.extend(audi_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Audi dataset: %d' % (len(audi_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = Path(__file__).resolve().parent / 'split' / (self.split + '.txt')
        self.sample_file_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def extract_image_file_name_from_lidar_file_name(self, file_name_lidar):
        file_name_image = file_name_lidar.split('/')
        file_name_image = file_name_image[-1].split('.')[0]
        file_name_image = file_name_image.split('_')
        file_name_image = file_name_image[0] + '_' + \
                            'camera_' + \
                            file_name_image[2] + '_' + \
                            file_name_image[3] + '.png'

        return file_name_image

    def extract_bboxes_file_name_from_image_file_name(self, file_name_image):
        """
            Let us read bounding boxes corresponding to the above image

        """
        file_name_bboxes = file_name_image.split('/')
        file_name_bboxes = file_name_bboxes[-1].split('.')[0]
        file_name_bboxes = file_name_bboxes.split('_')
        file_name_bboxes = file_name_bboxes[0] + '_' + \
                    'label3D_' + \
                    file_name_bboxes[2] + '_' + \
                    file_name_bboxes[3] + '.json'
        
        return file_name_bboxes

    def read_bounding_boxes(self, file_name_bboxes):
        """
        Read the bounding boxes corresponding to the frame. We can read the bounding boxes as follows
        """
        import json
        # open the file
        with open (file_name_bboxes, 'r') as f:
            bboxes = json.load(f)
            
        boxes = [] # a list for containing bounding boxes  
        print(bboxes.keys())
        
        for bbox in bboxes.keys():
            bbox_read = {} # a dictionary for a given bounding box
            bbox_read['class'] = bboxes[bbox]['class']
            bbox_read['truncation']= bboxes[bbox]['truncation']
            bbox_read['occlusion']= bboxes[bbox]['occlusion']
            bbox_read['alpha']= bboxes[bbox]['alpha']
            bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
            bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
            bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
            bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
            bbox_read['center'] =  np.array(bboxes[bbox]['center'])
            bbox_read['size'] =  np.array(bboxes[bbox]['size'])
            angle = bboxes[bbox]['rot_angle']
            axis = np.array(bboxes[bbox]['axis'])
            bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
            boxes.append(bbox_read)

        return boxes 

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_file_list=None):
        import concurrent.futures as futures

        def process_single_scene(file_name_lidar):
            # print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            file_name_lidar = str(self.root_path / file_name_lidar)
            seq_name = file_name_lidar.split('/')[-4]
            
            pc_info = {
                'lidar_path': file_name_lidar,
                'scene:': seq_name,
                'frame_name': file_name_lidar.split('/')[-1].replace('_lidar_frontcenter_','')
                }
            info['point_cloud'] = pc_info

            file_name_image = self.extract_image_file_name_from_lidar_file_name(file_name_lidar)
            file_name_image = join(self.root_path, 'camera_lidar_semantic_bboxes' ,seq_name, 'camera/cam_front_center/', file_name_image)
            # print("file_name_image:",file_name_image)
            image_info = {'image_path': file_name_image, 'scene': seq_name}
            info['image'] = image_info

            if has_label:
                file_name_bboxes = self.extract_bboxes_file_name_from_image_file_name(file_name_image)
                file_name_bboxes = join(self.root_path, 'camera_lidar_semantic_bboxes', seq_name, 'label3D/cam_front_center/', file_name_bboxes)
                with open (file_name_bboxes, 'r') as f:
                    bboxes = json.load(f)
                # print("boxes:",bboxes)

                annotations = {
                    'rotation': [],
                }
                # class \ truncation \ occlusion \ alpha \ 2d_bbox \ center \ size \ rot_angle \ axis
                for bbox in bboxes.keys():
                    for bbox_anno, bbox_anno_values in bboxes[bbox].items():
                        if bbox_anno in annotations:
                            annotations[bbox_anno].append(bbox_anno_values)
                        else:
                            annotations[bbox_anno] = [bbox_anno_values]
                    # gt_boxes = bboxes[bbox]['center'] + bboxes[bbox]['size'] + bboxes[bbox]['rot_angle']
                    annotations['rotation'].append(box_utils.axis_angle_to_rotation_mat(bboxes[bbox]['axis'], bboxes[bbox]['rot_angle']))
                
                for anno_key in annotations.keys():
                    annotations[anno_key] = np.array(annotations[anno_key])
                
                annotations['gt_boxes_lidar'] = np.hstack((annotations['center'], annotations['size'], annotations['rot_angle'].reshape(-1,1)))
                annotations['gt_boxes_lidar'][:,-1] = annotations['gt_boxes_lidar'][:,-1] * annotations['axis'][:,-1]
                # bbox3d_corner = box_utils.boxes_to_corners_3d(annotations['gt_boxes_lidar'])

                annotations["occluded"] = annotations["occlusion"]
                annotations["truncated"] = annotations["truncation"]
                # annotations["name"] = annotations["class"]
                annotations["name"] = []
                for i in range(annotations["class"].shape[0]):
                    class_name = annotations["class"][i]
                    if class_name == "Bicycle":
                        class_name = "Cyclist"
                    if class_name == "VanSUV":
                        class_name = "Car"
                    annotations["name"].append(class_name)
                annotations["name"] = np.array(annotations["name"])
                annotations.pop("occlusion")
                annotations.pop("truncation")
                # annotations.pop("class")
                annotations['dimensions'] = annotations['size']
                annotations['location'] = annotations['center']
                annotations['rotation_y'] = annotations['gt_boxes_lidar'][:,-1]
                annotations['bbox'] = annotations['2d_bbox']
                annotations.pop("size")
                annotations.pop("center")
                # annotations.pop("rot_angle")
                annotations.pop("2d_bbox")
                if count_inside_pts:
                    lidar_front_center = np.load(file_name_lidar)
                    points = lidar_front_center['points']

                    point_inbox_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points), torch.from_numpy(annotations['gt_boxes_lidar'][:,:7])
                    ).numpy()
                    annotations['num_points_in_gt'] = point_inbox_indices.sum(axis=-1)

                info['annos'] = annotations

            return info

        def clean_ignored_frame(sample_file_list):
            ignored_frame_file = Path(__file__).resolve().parent / 'split' / 'ignored_frame.txt'
            ignored_frame_list = [x.strip().split('/')[-1] for x in open(ignored_frame_file).readlines()] if ignored_frame_file.exists() else None
            cleaned_sample_file_list = []
            for sample_file in sample_file_list:
                if sample_file.split('/')[-1] not in ignored_frame_list:
                    cleaned_sample_file_list.append(sample_file)
            return cleaned_sample_file_list

        sample_file_list = sample_file_list if sample_file_list is not None else self.sample_file_list

        # sample_file_list = []

        # from os.path import join
        # for scene in sample_scene_list:
        #     file_names = sorted(glob.glob(join(self.root_path, 'camera_lidar_semantic_bboxes/', scene+'/', 'lidar/cam_front_center/*.npz')))
        #     sample_file_list += file_names
            
        print("sample_file_list:",len(sample_file_list))
        sample_file_list = clean_ignored_frame(sample_file_list)
        print("sample_file_list after clean ignored frames:",len(sample_file_list))

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_file_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('audi_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['lidar_idx']
            # points = self.get_lidar(sample_idx, num_features=info['point_cloud']['num_features'])
            
            lidar_path = info['point_cloud']['lidar_path']
            lidar_front_center = np.load(lidar_path)
            
            points = lidar_front_center['points']
            reflectance = lidar_front_center['reflectance']/255.0
            lidar_id = lidar_front_center['lidar_id']
            points = np.hstack((points, reflectance.reshape(-1,1), lidar_id.reshape(-1,1)))
                    
            if self.dataset_cfg.ONE_LIDAR_POINTS_ONLY:
                points = points[lidar_id==3]
            
            annos = info['annos']
            names = annos['name']
            difficulty = annos['occluded']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points = gt_points.astype(np.float32)
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': 1}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index]
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = np.zeros(pred_scores.shape[0])
            pred_dict['bbox'] = np.ones((pred_scores.shape[0],4))
            # pred_dict['dimensions'] = pred_boxes[:, 3:6]
            # x,y,z = pred_boxes[:, 0:1], pred_boxes[:, 1:2], pred_boxes[:, 2:3]
            # l, w, h, r = pred_boxes[:, 3:4], pred_boxes[:, 4:5], pred_boxes[:, 5:6], pred_boxes[:, 6:7]
            # pred_dict['dimensions'] =  np.concatenate([l, h, w], axis=-1)
            # pred_dict['location'] = np.concatenate([-y, x, z], axis=-1)
            # pred_dict['rotation_y'] = pred_boxes[:, 6]

            x,y,z = pred_boxes[:, 0:1], pred_boxes[:, 1:2], pred_boxes[:, 2:3]
            l, w, h, r = pred_boxes[:, 3:4], pred_boxes[:, 4:5], pred_boxes[:, 5:6], pred_boxes[:, 6:7]
            z -= h/2
            pred_dict['dimensions'] = np.concatenate([l, h, w], axis=-1)
            pred_dict['location'] = np.concatenate([-y, -z, x], axis=-1)
            pred_dict['rotation_y'] = -np.pi/2 - pred_boxes[:, 6]

            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.audi_infos[0].keys():
            return None, {}

        from .audi_object_eval_python import eval as audi_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.audi_infos]

        for i in range(len(eval_gt_annos)):
            # boxes3d = eval_gt_annos[i]['gt_boxes_lidar']
            # x,y,z = boxes3d[:, 0:1], boxes3d[:, 1:2], boxes3d[:, 2:3]
            # l, w, h, r = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
            # eval_gt_annos[i]['dimensions'] = np.concatenate([l, h, w], axis=-1)
            # eval_gt_annos[i]['location'] = np.concatenate([-y, x, z], axis=-1)
            
            boxes3d = eval_gt_annos[i]['gt_boxes_lidar']
            x,y,z = boxes3d[:, 0:1], boxes3d[:, 1:2], boxes3d[:, 2:3]
            l, w, h, r = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
            z -= h/2
            eval_gt_annos[i]['dimensions'] = np.concatenate([l, h, w], axis=-1)
            eval_gt_annos[i]['location'] = np.concatenate([-y, -z, x], axis=-1)
            eval_gt_annos[i]['rotation_y'] = -np.pi/2 - boxes3d[:, 6]

            num_points_in_gt = eval_gt_annos[i]['num_points_in_gt']
            mask = num_points_in_gt > 0
            eval_gt_annos[i]['dimensions'] = eval_gt_annos[i]['dimensions'][mask]
            eval_gt_annos[i]['location'] = eval_gt_annos[i]['location'][mask]
            eval_gt_annos[i]['rotation_y'] = eval_gt_annos[i]['rotation_y'][mask]
            eval_gt_annos[i]['gt_boxes_lidar'] = eval_gt_annos[i]['gt_boxes_lidar'][mask]
            eval_gt_annos[i]['occluded'] = eval_gt_annos[i]['occluded'][mask]
            eval_gt_annos[i]['truncated'] = eval_gt_annos[i]['truncated'][mask]
            eval_gt_annos[i]['alpha'] = eval_gt_annos[i]['alpha'][mask]
            eval_gt_annos[i]['bbox'] = eval_gt_annos[i]['bbox'][mask]
            eval_gt_annos[i]['name'] = eval_gt_annos[i]['name'][mask]

            # eval_gt_annos[i]['rotation_y'] = -np.pi/2 - boxes3d[:, 6]
            # eval_gt_annos[i]['gt_boxes_lidar'] = np.concatenate([-y, x, z, l, h, w, r], axis=-1)

        # num = 2
        # eval_det_annos = eval_det_annos[:num]
        # eval_gt_annos = eval_gt_annos[:num]
        
        if 'ignore_classes' in kwargs and kwargs['ignore_classes']:
            # Pedestrian,Cyclist --> Car
            for det_anno in eval_det_annos:
                det_anno_name = []
                for i in range(det_anno['name'].shape[0]):
                    if det_anno['name'][i] != 'Car':
                        name = 'Car'
                    else:
                        name = det_anno['name'][i]
                    det_anno_name.append(name)
                det_anno['name'] = np.array(det_anno_name) 

        ap_result_str, ap_dict = audi_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names, compute_cls_ap=True, PR_detail_dict={})
        

        return ap_result_str, ap_dict

    def get_detobject_iou(self, det_annos, **kwargs):
        if 'annos' not in self.audi_infos[0].keys():
            return None, {}

        from .audi_object_eval_python import eval as audi_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.audi_infos]
        
        for i in range(len(eval_gt_annos)):
            boxes3d = eval_gt_annos[i]['gt_boxes_lidar']
            x,y,z = boxes3d[:, 0:1], boxes3d[:, 1:2], boxes3d[:, 2:3]
            l, w, h, r = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
            z -= h/2
            eval_gt_annos[i]['dimensions'] = np.concatenate([l, h, w], axis=-1)
            eval_gt_annos[i]['location'] = np.concatenate([-y, -z, x], axis=-1)
            eval_gt_annos[i]['rotation_y'] = -np.pi/2 - boxes3d[:, 6]

        # num = 2
        rets = audi_eval.calculate_iou_partly(eval_det_annos, eval_gt_annos, metric=2, num_parts=100)
        # print("eval_det_annos:",eval_det_annos[:num])
        # print("eval_gt_annos:",eval_gt_annos[:num])
        # print("rets:",rets)
        # print("rets0:",rets[0])
        # print("-----------")
        for i in range(len(eval_det_annos)):
            ret = rets[0][i]
            # idx = ret.argmax(axis=-1)
            iou = ret.max(axis=-1)
            eval_det_annos[i]['iou'] = iou
            # print(i,"\n",eval_det_annos[i])
            # print("ret:",ret)
            # print("idx:",idx)
            # print("iou:",iou)

        return eval_det_annos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.audi_infos) * self.total_epochs

        return len(self.audi_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.audi_infos)

        info = copy.deepcopy(self.audi_infos[index])

        lidar_idx = info['lidar_idx']
        lidar_path = info['point_cloud']['lidar_path']
        lidar_front_center = np.load(lidar_path)
        
        points = lidar_front_center['points']
        reflectance = lidar_front_center['reflectance']/255.0
        lidar_id = lidar_front_center['lidar_id']
        points = np.hstack((points, reflectance.reshape(-1,1), lidar_id.reshape(-1,1)))
        
        if self.dataset_cfg.ONE_LIDAR_POINTS_ONLY:
            points = points[lidar_id==3]

        input_dict = {
            'points': points,
            'frame_id': info['lidar_idx'],
            'metadata': {
                'data_type':'audi',
                'frame_name': info['point_cloud']['frame_name']
                },
        } # elodie metadata

        if 'annos' in info:
            annos = info['annos']
            
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['gt_boxes_lidar']
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_audi_infos_back(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('audi_infos_%s.pkl' % train_split)
    val_filename = save_path / ('audi_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'audi_infos_trainval.pkl'

    print('---------------Start to generate Audi data infos---------------')

    dataset.set_split(train_split)
    audi_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    for i in range(len(audi_infos_train)):
        audi_infos_train[i]['lidar_idx'] = i

    with open(train_filename, 'wb') as f:
        pickle.dump(audi_infos_train, f)
    print('Audi info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    audi_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    for i in range(len(audi_infos_val)):
        audi_infos_val[i]['lidar_idx'] = len(audi_infos_train) + i

    with open(val_filename, 'wb') as f:
        pickle.dump(audi_infos_val, f)
    print('Audi info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(audi_infos_train + audi_infos_val, f)
    print('Audi info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    audi_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(audi_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

def create_audi_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = AudiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    print("dataset.root_path:",dataset.root_path)
    train_filename = save_path / ('audi_infos_%s.pkl' % train_split)
    val_filename = save_path / ('audi_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'audi_infos_trainval.pkl'

    # print('---------------Start to generate Audi data infos---------------')

    dataset.set_split(train_split)
    audi_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    for i in range(len(audi_infos_train)):
        audi_infos_train[i]['lidar_idx'] = i
    with open(train_filename, 'wb') as f:
        pickle.dump(audi_infos_train, f)
    print('Audi info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    audi_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    for i in range(len(audi_infos_val)):
        audi_infos_val[i]['lidar_idx'] = len(audi_infos_train) + i
    with open(val_filename, 'wb') as f:
        pickle.dump(audi_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(audi_infos_train + audi_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # audi_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(audi_infos_test, f)
    # print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


# if __name__ == '__main__':
#     import sys
#     if sys.argv.__len__() > 1 and sys.argv[1] == 'create_audi_infos':
#         import yaml
#         from pathlib import Path
#         from easydict import EasyDict
#         dataset_cfg = EasyDict(yaml.load(open(sys.argv[2]),Loader=yaml.FullLoader)) # elodie
#         ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
#         print("ROOT_DIR:",ROOT_DIR)
#         create_audi_infos(
#             dataset_cfg=dataset_cfg,
#             class_names=['Car', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'kitti',
#             save_path=ROOT_DIR / 'data' / 'kitti'
#         )

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1:
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[1]),Loader=yaml.FullLoader)) # elodie
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        print("ROOT_DIR:",ROOT_DIR)
        class_names=['Car', 'Pedestrian', 'Cyclist']

        create_audi_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'audi_mimic',
            save_path=ROOT_DIR / 'data' / 'audi_mimic'
        )
