from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils, pointcloud_sample_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

import copy
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, "Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features, root_path=None):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.root_path = root_path

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('dense', None) is not None:
            mask_dense = common_utils.mask_points_by_range(data_dict['dense']['points_dense'], self.point_cloud_range)
            data_dict['dense']['points_dense'] = data_dict['dense']['points_dense'][mask_dense]
        if 'ring' in data_dict:
            data_dict['ring'] = data_dict['ring'][mask]
        if 'lidar_id' in data_dict:
            data_dict['lidar_id'] = data_dict['lidar_id'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            if 'gt_obj_ids' in data_dict:
                data_dict['gt_obj_ids'] = data_dict['gt_obj_ids'][mask]
        return data_dict

    def shuffle_points_unit(self, points):
        shuffle_idx = np.random.permutation(points.shape[0])
        return points[shuffle_idx]
        
    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        mimic_shuffle = config.get('MIMIC_SHUFFLE_ENABLED', None)
        if mimic_shuffle is not None and mimic_shuffle[self.mode]:
            if '16lines' in data_dict: #elodie
                data_dict['16lines']['points_16lines'] = self.shuffle_points_unit(data_dict['16lines']['points_16lines'])
            if 'dense' in data_dict: #elodie
                data_dict['dense']['points_dense'] = self.shuffle_points_unit(data_dict['dense']['points_dense'])

        if "16lines" in data_dict and "extra_points_16lines" in data_dict["16lines"]:
            if config.SHUFFLE_ENABLED[self.mode]:
                extra_points = data_dict['16lines']['extra_points_16lines']
                shuffle_idx = np.random.permutation(extra_points.shape[0])
                extra_points = extra_points[shuffle_idx]
                data_dict['points'] = np.vstack((data_dict['16lines']['points_16lines'], extra_points))
            else:
                data_dict['points'] = np.vstack((data_dict['16lines']['points_16lines'], data_dict['16lines']['extra_points_16lines']))
            data_dict['16lines'].pop('extra_points_16lines')
        else:
            if config.SHUFFLE_ENABLED[self.mode]:
                points = data_dict['points']
                shuffle_idx = np.random.permutation(points.shape[0])
                points = points[shuffle_idx]
                data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_mimic(self, mimic_key, data_dict=None):
        if mimic_key in data_dict:
            points_name = "points_%s"%(mimic_key)
            points_mimic = data_dict[mimic_key][points_name]
            voxel_output_mimic = self.voxel_generator.generate(points_mimic)
            if isinstance(voxel_output_mimic, dict):
                voxels_mimic, coordinates_mimic, num_points_mimic = \
                    voxel_output_mimic['voxels'], voxel_output_mimic['coordinates'], voxel_output_mimic['num_points_per_voxel']
            else:
                voxels_mimic, coordinates_mimic, num_points_mimic = voxel_output_mimic
            
            if not data_dict['use_lead_xyz']:
                voxels_mimic = voxels_mimic[..., 3:]  # remove xyz in voxels(N, 3)

            data_dict[mimic_key]['voxels'] = voxels_mimic
            data_dict[mimic_key]['voxel_coords'] = coordinates_mimic
            data_dict[mimic_key]['voxel_num_points'] = num_points_mimic

            if 'points_16lines_inbox' in data_dict[mimic_key]:
                voxel_output_mimic_inbox = self.voxel_generator.generate(data_dict[mimic_key]['points_16lines_inbox'])
                if isinstance(voxel_output_mimic, dict):
                    data_dict[mimic_key]['voxel_coords_inbox'] = voxel_output_mimic_inbox['coordinates']
                else:
                    data_dict[mimic_key]['voxel_coords_inbox'] = voxel_output_mimic_inbox[1]
    
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        
        if '16lines' in data_dict: #elodie
            self.transform_points_to_voxels_mimic(mimic_key='16lines', data_dict=data_dict)
        if 'dense' in data_dict: #elodie
            self.transform_points_to_voxels_mimic(mimic_key='dense', data_dict=data_dict)
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    # @brief: downsample pointcloud to 16 lines - elodie
    def downsample_points_16lines(self, data_dict=None, config=None): 
        if data_dict is None:
            return partial(self.downsample_points_16lines, config=config)
        assert "data_type" in data_dict["metadata"], '[Error Elodie] metadata not in data_dict!'
        assert "ring" in data_dict, '[Error Elodie] ring not in data_dict!'
        if config.DOWNSAMPLE_POINTS[self.mode] is not True:
            data_dict.pop('ring')
            return data_dict

        downsample_type = config.get('DOWNSAMPLE_TYPE', 'TensorPro')
        assert downsample_type in ['VLP16','TensorPro', 'TensorPro_v2','TensorPro_v42', 'TensorPro_v5', 'Waymo_v1', 'Waymo_v2', 'Waymo_v3', 'Waymo_64'], '[Error Elodie] DOWNSAMPLE_TYPE is neither TensorPro nor VLP16!'
        align_points_switch = config.get('ALIGN_POINTS', False)
        verticle_switch = config.get('VERTICAL_SAMPLE', True)
        horizontal_switch = config.get('HORIZONTAL_SAMPLE', True)

        points = data_dict['points']
        data_type = data_dict["metadata"]["data_type"]

        if data_type == "kitti":
            if downsample_type == "TensorPro":
                points_16lines, extra_points = pointcloud_sample_utils.downsample_kitti(points, data_dict['ring'], verticle_switch=verticle_switch, horizontal_switch=horizontal_switch, return_extra_points=align_points_switch)
            elif downsample_type == "TensorPro_v2":
                points_16lines = pointcloud_sample_utils.downsample_kitti_v2(points, data_dict['ring'], verticle_switch=verticle_switch, horizontal_switch=horizontal_switch)
            elif downsample_type == "TensorPro_v42":
                points_16lines = pointcloud_sample_utils.downsample_kitti_v42(points, data_dict['ring'], verticle_switch=verticle_switch)
            elif downsample_type == "TensorPro_v5":
                points_16lines = pointcloud_sample_utils.downsample_kitti_v5(points, data_dict['ring'], verticle_switch=verticle_switch)
            elif downsample_type == "VLP16":
                points_16lines, extra_points = pointcloud_sample_utils.downsample_kitti_to_VLP16(points, data_dict['ring'], verticle_switch=verticle_switch, return_extra_points=align_points_switch)
        elif data_type == "nuscenes":
            points_16lines = pointcloud_sample_utils.downsample_nusc_v2(points, data_dict['ring'])
            points_16lines = pointcloud_sample_utils.upsample_nusc_v1(points_16lines, data_dict['ring'])
        elif data_type == "waymo":
            points_16lines = pointcloud_sample_utils.downsample_waymo(points, data_dict['ring'], sample_type=downsample_type)
        if config.REPLACE_ORI_POINTS[self.mode]:
            data_dict['points'] = points_16lines
        else:
            data_dict['16lines'] = {}
            data_dict['16lines']['points_16lines'] = points_16lines
            if align_points_switch: # elodie : if align_points is False, extra_points will be None 
                data_dict['16lines']['extra_points_16lines'] = extra_points
            if config.get('GET_INBOX_POINTS', False): #elodie
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points_16lines[:, 0:3]), torch.from_numpy(data_dict['gt_boxes'][:,:7])
                ).numpy()
                data_dict['16lines']['points_16lines_inbox'] = points_16lines[point_indices.sum(axis=0) == 1]
        data_dict.pop('ring')
        return data_dict

    # @brief: select pointclouds to 16 lines by lidar id for Audi Dataset - elodie
    def select_points_by_lidar_id(self, data_dict=None, config=None): 
        if data_dict is None:
            return partial(self.select_points_by_lidar_id, config=config)
        assert "lidar_id" in data_dict, '[Error Elodie] lidar_id not in data_dict!'

        if config.DOWNSAMPLE_POINTS[self.mode] is not True:
            data_dict.pop('lidar_id')
            return data_dict

        align_points_switch = config.get('ALIGN_POINTS', False)

        points = data_dict['points']
        lidar_id = data_dict['lidar_id']

        if config.REPLACE_ORI_POINTS[self.mode]:
            data_dict['points'] = points[lidar_id==3]
        else:
            points_16lines = points[lidar_id==3]
            data_dict['16lines'] = {}
            data_dict['16lines']['points_16lines'] = points_16lines
            if align_points_switch: # elodie : if align_points is False, extra_points will be None 
                data_dict['16lines']['extra_points_16lines'] = points[lidar_id!=3]
            if config.get('GET_INBOX_POINTS', False): #elodie
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points_16lines[:, 0:3]), torch.from_numpy(data_dict['gt_boxes'][:,:7])
                ).numpy()
                data_dict['16lines']['points_16lines_inbox'] = points_16lines[point_indices.sum(axis=0) == 1]
        data_dict.pop('lidar_id')
        return data_dict

    def mask_bboxed_by_min_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_bboxed_by_min_points, config=config)
        # assert "16lines" in data_dict, '[Error Elodie] 16lines not in data_dict!'
        if config.MASK_ENABLED[self.mode]:
            gt_boxes_lidar = data_dict['gt_boxes']
            if config.USE_SAMPLE_POINTS[self.mode]:
                points_16lines = data_dict['16lines']['points_16lines']
            else:
                points_16lines = data_dict['points']
            num_objects = gt_boxes_lidar.shape[0]
            corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
            num_points_in_gt = -np.ones(num_objects, dtype=np.int32)

            for k in range(num_objects):
                flag = box_utils.in_hull(points_16lines[:, 0:3], corners_lidar[k])
                num_points_in_gt[k] = flag.sum()
            # data_dict['num_points_in_gt_sampled'] = num_points_in_gt
            mask = num_points_in_gt > 0 
            data_dict['gt_boxes'] = gt_boxes_lidar[mask]
        return data_dict

    def complish_gt_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.complish_gt_points, config=config)
        assert "gt_obj_ids" in data_dict, '[Error Elodie] gt_obj_ids not in data_dict!'
        
        if config.COMPLISH_ENABLED[self.mode]:
            points = data_dict['points']
            # points = box_utils.remove_points_in_boxes3d(points, data_dict['gt_boxes'])
            completed_points_all = None
            for box, obj_id_key,num_points_in_gt in zip(data_dict['gt_boxes'], data_dict['gt_obj_ids'],data_dict['num_points_in_gt']):
                # if num_points_in_gt > config.NUM_POINTS_GT:
                    # continue
                try:
                    filename = '%s.bin' % (obj_id_key)
                    file_path = self.root_path /  config.DIRNAME / filename
                    completed_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                            [-1, config.NUM_POINT_FEATURES])
                    if config.RANDOM_SAMPLE:
                        num_points = config.NUM_POINTS
                        if num_points < completed_points.shape[0]:
                            choice = np.arange(0, completed_points.shape[0], dtype=np.int32)
                            choice = np.random.choice(choice, num_points, replace=False)
                            completed_points = completed_points[choice]
                    completed_points = common_utils.rotate_points_along_z(completed_points[np.newaxis, :, :], np.array([box[6]]))[0]
                    completed_points[:, :3] += box[:3]
                    if completed_points_all is None:
                        completed_points_all = completed_points
                    else:
                        completed_points_all = np.vstack((completed_points_all, completed_points))
                except:
                    pass
            if completed_points_all is not None:
                points_compt = np.concatenate([completed_points_all, points], axis=0)
            else:
                points_compt = points

            if config.REPLACE_ORI_POINTS[self.mode]:
                data_dict['points'] = points_compt
            else:
                data_dict['dense'] = {}
                data_dict['dense']['points_dense'] = points_compt
        return data_dict
    
    def concat_completed_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.concat_completed_points, config=config)
        
        if config.CONCAT_ENABLED[self.mode]:
            assert "dense" in data_dict, '[Error Elodie] dense not in data_dict!'
            assert "points_ori_num" in data_dict["dense"], '[Error Elodie] points_ori_num not in data_dict[dense]!'
            points = data_dict['points']
            points_ori_num = data_dict['dense']['points_ori_num']
            data_dict['dense']['points_dense'] = points
            data_dict['points'] = points[:points_ori_num]

            if 'ring' in data_dict:
                data_dict['ring'] = data_dict['ring'][:points_ori_num]
        return data_dict


    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
