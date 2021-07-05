from functools import partial

import numpy as np

from ...utils import common_utils, box_utils
from . import augmentor_utils, database_sampler

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def complish_gt_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.complish_gt_points, config=config)
        assert "gt_obj_ids" in data_dict, '[Error Elodie] gt_obj_ids not in data_dict!'
        
        points = data_dict['points']
        completed_gt_dirname = config.DIRNAME
        # points = box_utils.remove_points_in_boxes3d(points, data_dict['gt_boxes'])
        completed_points_all = None
        for box, obj_id_key, num_points_in_gt in zip(data_dict['gt_boxes'], data_dict['gt_obj_ids'], data_dict['num_points_in_gt']):
            if num_points_in_gt > config.NUM_POINTS_GT:
                continue
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
                completed_points = common_utils.rotate_points_along_z(completed_points[np.newaxis, :, :], np.array([box[-1]]))[0]
                completed_points[:, :3] += box[:3]
                if completed_points_all is None:
                    completed_points_all = completed_points
                else:
                    completed_points_all = np.vstack((completed_points_all, completed_points))
            except:
                pass
        points = np.concatenate([completed_points_all, points], axis=0)
        data_dict['points'] = points

        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
