import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils, mimic_loss_utils


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True, cls_score_thred=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        distill_switch = self.build_soft_losses()
        self.build_losses(return_weights=distill_switch)


    def build_losses(self, return_weights=False):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet(return_weights=return_weights))
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    # for distillation
    def build_soft_losses(self):
        soft_losses_cfg = self.model_cfg.get('SOFT_LOSS_CONFIG', None) # elodie soft loss
        self.cls_score_thred = self.model_cfg.LOSS_CONFIG.get('CLS_SCORE_THRED', \
                                            self.model_cfg.POST_PROCESSING.SCORE_THRESH)
        self.soft_loss_weights = {}
        self.cls_soft_loss_type = None
        self.reg_soft_loss_type = None
        self.dir_soft_loss_type = None
        self.hint_soft_loss_type = None
        self.mimic_cls_classes_use_only = False

        if soft_losses_cfg is None:
            return False

        if soft_losses_cfg.get('CLS_LOSS', None) is not None:
            self.cls_soft_loss_type = soft_losses_cfg.CLS_LOSS.TYPE
            mimic_cls_temperature = soft_losses_cfg.CLS_LOSS.get('TEMPERATURE', 1.0)
            if self.cls_soft_loss_type in ['SigmoidFocalClassificationLoss', 'SigmoidFocalLoss']:
                self.add_module(
                    'soft_hm_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
                    )
            elif self.cls_soft_loss_type in ['WeightedKLDivergenceLoss_v2']:
                weighted = soft_losses_cfg.CLS_LOSS.get('WEIGHTED', True)
                self.mimic_cls_classes_use_only = soft_losses_cfg.CLS_LOSS.get('CLASS_USE_ONLY', False)
                if self.mimic_cls_classes_use_only: # elodie
                    class_index = []
                    num_orient = sum(self.num_anchors_per_location) // self.num_class
                    for i in range(self.num_class):
                        for j in range(num_orient):
                            class_index.append((i*2+j)*self.num_class +i)
                    self.class_index = np.array(class_index)
                self.add_module(
                    'soft_hm_loss_func',
                     getattr(mimic_loss_utils, self.cls_soft_loss_type)(weighted=weighted, \
                                                        T=mimic_cls_temperature, activated=True)
                    )
            else:
                self.add_module(
                    'soft_hm_loss_func',
                    getattr(loss_utils, self.cls_soft_loss_type)()
                )
            self.cls_soft_loss_beta = soft_losses_cfg.CLS_LOSS.get('BETA', 0.5)
            self.cls_soft_loss_modify = soft_losses_cfg.CLS_LOSS.get('MODIFY', None)
            self.cls_soft_loss_source = soft_losses_cfg.CLS_LOSS.get('SOURCE', None)
            self.cls_use_teacher_t_only = soft_losses_cfg.CLS_LOSS.get('ONLY_USE_TRUE_RET', False)
            self.cls_target_ignore = soft_losses_cfg.CLS_LOSS.get('TARGET_IGNORE', False)
            self.cls_soft_loss_source_weights = soft_losses_cfg.CLS_LOSS.get('SOURCE_WEIGHTS', None)
            if self.cls_soft_loss_source_weights is None and self.cls_soft_loss_source is not None:
                self.cls_soft_loss_source_weights = np.ones(len(self.cls_soft_loss_source))

        if soft_losses_cfg.get('REG_LOSS', None) is not None:
            self.reg_soft_loss_type = soft_losses_cfg.REG_LOSS.TYPE
            self.add_module(
                'soft_reg_loss_func',
                getattr(mimic_loss_utils, self.reg_soft_loss_type)(\
                    margin=soft_losses_cfg.REG_LOSS.get('MARGIN', 0.001))
            )
            self.reg_soft_loss_alpha = soft_losses_cfg.REG_LOSS.get('ALPHA', 0.5)
            self.reg_soft_loss_modify = soft_losses_cfg.REG_LOSS.get('MODIFY', None)
            self.reg_soft_loss_source = soft_losses_cfg.REG_LOSS.get('SOURCE', None)
            self.reg_soft_loss_source_weights = soft_losses_cfg.REG_LOSS.get('SOURCE_WEIGHTS', None)
            if self.reg_soft_loss_source_weights is None and self.reg_soft_loss_source is not None:
                self.reg_soft_loss_source_weights = np.ones(len(self.reg_soft_loss_source))
            self.reg_soft_loss_use_sin = soft_losses_cfg.REG_LOSS.get('USE_SIN', False)

        if soft_losses_cfg.get('HINT_LOSS', None) is not None:
            self.hint_soft_loss_type = soft_losses_cfg.HINT_LOSS.TYPE

            hint_soft_loss_temperature = soft_losses_cfg.HINT_LOSS.get('TEMPERATURE', 1)
            hint_normalize = soft_losses_cfg.HINT_LOSS.get('NORMALIZE', True)

            self.add_module(
                'soft_hint_loss_func',
                getattr(mimic_loss_utils, self.hint_soft_loss_type)(T=hint_soft_loss_temperature, normalize=hint_normalize)
            )
            self.hint_soft_loss_gamma = soft_losses_cfg.HINT_LOSS.get('GAMMA', 0.5)
            self.hint_feature_list = soft_losses_cfg.HINT_LOSS.get('FEATURE_LIST', None)
            self.hint_feature_weights = soft_losses_cfg.HINT_LOSS.get('FEATURE_WEIGHTS', None)
            if self.hint_feature_weights is None:
                self.hint_feature_weights = np.ones(len(self.hint_feature_list))
            self.seg_batch = soft_losses_cfg.HINT_LOSS.get('SEG_BATCH', False)
            self.hint_soft_loss_source = soft_losses_cfg.HINT_LOSS.get('SOURCE', None)
            self.hint_soft_loss_source_weights = soft_losses_cfg.HINT_LOSS.get('SOURCE_WEIGHTS', None)
            if self.hint_soft_loss_source_weights is None and self.hint_soft_loss_source is not None:
                self.hint_soft_loss_source_weights = np.ones(len(self.hint_soft_loss_source))

        return True


    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['hm_hard_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            tb_dict['loc_hard_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    # for distillation
    def get_loss(self, teacher_ret_dict=None, student_data_dict=None, teacher_data_dict=None):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        if teacher_ret_dict is not None:
            teacher_ret_dict = teacher_ret_dict['pred_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss, focal_weights = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            # distillation
            if teacher_ret_dict is not None and self.cls_soft_loss_type is not None:
                teacher_preds = teacher_ret_dict[idx]
                teacher_preds['hm'] = self.sigmoid(teacher_preds['hm'])
                batch_size = pred_dict['hm'].shape[0]

                self.soft_loss_weights['weights_focal'] = focal_weights.sum(dim=1)
                # ========= hm soft loss ==========
                # hm_soft_weights = self.get_soft_loss_weights(pred_dict['hm'].permute(0, 2, 3, 1), \
                #                                             teacher_preds['hm'].permute(0, 2, 3, 1), \
                #                                             target_dicts['heatmaps'][idx].permute(0, 2, 3, 1)
                #                                             )
                if self.cls_soft_loss_source is None:
                    hm_soft_weights = self.soft_loss_weights['weights_focal']
                else:
                    hm_soft_weights = torch.full_like(self.soft_loss_weights['weights_focal'], 0, \
                        dtype=self.soft_loss_weights['weights_focal'].dtype)
                    for src, src_weights in zip(self.cls_soft_loss_source, self.cls_soft_loss_source_weights):
                        if src == "FocalWeight":
                            hm_soft_weights += src_weights*self.soft_loss_weights['weights_focal']
                    
                hm_soft_loss = self.soft_hm_loss_func(pred_dict['hm'].permute(0, 2, 3, 1), \
                                                            teacher_preds['hm'].permute(0, 2, 3, 1), \
                                                            hm_soft_weights
                                                            )
                
                # for debug print
                # for i in range(188):
                #     for j in range(188):
                #         # if target_dicts['heatmaps'][idx][0,:, i, j].sum()==0 and (teacher_preds['hm'][0, :, i, j]<0.1).float().sum()>0:
                #             # continue
                #         print("-------", i, j, "-------")
                #         print("\t", i, j, "- target_dicts['heatmaps'][idx]",target_dicts['heatmaps'][idx][0,:, i, j])
                #         print("\t", i, j,  "- teacher_preds['hm']", teacher_preds['hm'][0, :, i, j])
                #         print("\t", i, j, "- pred_dict['hm']", pred_dict['hm'][0,:, i, j])
                #         print("\t", i, j, "- focal_weights", focal_weights[0,:, i, j])
                #         print("\t", i, j, "- hm_soft_loss", hm_soft_loss[0, i, j])

                # =================for debug print

                cls_soft_loss = self.cls_soft_loss_beta * hm_soft_loss.sum() / batch_size

                
                # if self.cls_soft_loss_modify is not None:
                #     hm_loss = (1-self.cls_soft_loss_modify)*hm_loss + self.cls_soft_loss_modify * cls_soft_loss
                # else:
                #     hm_loss = hm_loss + cls_soft_loss

                # ========== regression soft loss ==========
                teacher_pred_boxes = torch.cat([teacher_preds[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

                if self.reg_soft_loss_source is not None:
                    reg_soft_weights = torch.full_like(self.soft_loss_weights['weights_focal'][:,None,...], 0, \
                        dtype=self.soft_loss_weights['weights_focal'].dtype)
                    for src, src_weights in zip(self.reg_soft_loss_source, self.reg_soft_loss_source_weights):
                        if src == "FocalGTWeight":
                            reg_soft_weights += src_weights * self.soft_loss_weights['weights_focal'][:,None,...]
                else:
                    reg_soft_weights = self.soft_loss_weights['weights_focal'][:,None,...]
                    
                regr, teach_regr, gt_regr, reg_weights = self.trans_reg_pred(pred_boxes, teacher_pred_boxes, \
                                                target_dicts['masks'][idx], \
                                                target_dicts['inds'][idx], \
                                                target_boxes, \
                                                weights_soft = reg_soft_weights,
                                                )

                loc_soft_loss_src = self.soft_reg_loss_func(regr, teach_regr, gt_regr, \
                                                            weights=reg_weights)

                # for debug print
                # for i in range(gt_regr.shape[1]):
                #         # if target_dicts['heatmaps'][idx][0,:, i, j].sum()==0 and (teacher_preds['hm'][0, :, i, j]<0.1).float().sum()>0:
                #             # continue
                #         print("-------", i,  "-------")
                #         print("\t", i,  "- gt_regr",gt_regr[0,i])
                #         print("\t", i,  "- teach_regr",teach_regr[0,i])
                #         print("\t", i,  "- regr",regr[0,i])
                #         print("\t", i,  "- reg_weights",reg_weights[0,i])
                #         print("\t", i,  "- loc_soft_loss_src",loc_soft_loss_src[0,i])

                # =================for debug print
                loc_soft_loss = self.reg_soft_loss_alpha *loc_soft_loss_src.sum() / batch_size
                
                # ========== hint soft loss  ==========
                if self.hint_soft_loss_type is not None:
                    hint_loss, tb_dict_hint = self.get_hint_loss(student_data_dict=student_data_dict, teacher_data_dict=teacher_data_dict)
                    loss += hint_loss
                    tb_dict.update(tb_dict_hint)
                
                tb_dict_soft = {
                    'hm_hard_loss_head_%d' % idx: copy.deepcopy(hm_loss.item()),
                    'hm_soft_loss_%d' % idx: cls_soft_loss.item(),
                    'loc_hard_loss_head_%d' % idx: copy.deepcopy(loc_loss.item()),
                    'loc_soft_loss%d' % idx: loc_soft_loss.item(),
                }
                tb_dict.update(tb_dict_soft)
                loc_soft_loss = loc_soft_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                loss += cls_soft_loss + loc_soft_loss  

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict
    
    def get_hint_loss(self, student_data_dict=None, teacher_data_dict=None):
        hint_loss = 0.0
        if self.hint_soft_loss_source is None:
            weights = None
        else:
            # weights = torch.full_like(self.soft_loss_weights['weights_gt'], 0, dtype=self.soft_loss_weights['weights_gt'].dtype)
            for src, src_loss_weights in zip(self.hint_soft_loss_source, self.hint_soft_loss_source_weights):
                if src == "FocalWeight":
                    B = self.soft_loss_weights['weights_focal'].size(0)
                    weights = self.soft_loss_weights['weights_focal'].view(B, -1)
                    weights = src_loss_weights * weights

        assert len(self.hint_feature_list) == len(self.hint_feature_weights), 'self.hint_feature_list length != self.hint_feature_weights length'
        for i, feature_ in enumerate(self.hint_feature_list):
                teacher_feature = teacher_data_dict[feature_]
                student_feature = student_data_dict[feature_]
                student_feature = student_feature.permute(0, 2, 3, 1) # [N,H,W,C]
                student_feature = student_feature.view(student_feature.shape[0], -1, student_feature.shape[-1])

                teacher_feature = teacher_feature.permute(0, 2, 3, 1) # [N,H,W,C]
                teacher_feature = teacher_feature.view(teacher_feature.shape[0], -1, teacher_feature.shape[-1])

                batch_size = int(student_feature.shape[0])
                if weights is not None:
                    hint_loss_src = self.soft_hint_loss_func(student_feature,teacher_feature,weights=weights)
                    hint_loss_src = hint_loss_src.sum()/batch_size
                    # print("hint_loss_src:",hint_loss_src)
                else:
                    hint_loss_src = self.soft_hint_loss_func(student_feature,teacher_feature)
                    hint_loss_src = hint_loss_src.mean()
                
                hint_loss = hint_loss + self.hint_soft_loss_gamma * hint_loss_src * self.hint_feature_weights[i]

        tb_dict = {
            'hint_loss': hint_loss.item()
        }
        return hint_loss, tb_dict

    def trans_reg_pred(self, output, teach_output, mask, ind=None, target=None, weights_soft=None):
        """
        from loss_utils
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
            teach_pred = teach_output
            weights_soft_ind = weights_soft
        else:
            pred = loss_utils._transpose_and_gather_feat(output, ind)
            teach_pred = loss_utils._transpose_and_gather_feat(teach_output, ind)
            weights_soft_ind = loss_utils._transpose_and_gather_feat(weights_soft, ind)
        
        num = mask.float().sum()
        weights = mask.float() / mask.float().sum(dim=1)[...,None]

        mask = mask.unsqueeze(2).expand_as(target).float()

        isnotnan = (~ torch.isnan(target)).float()
        mask *= isnotnan
        regr = pred * mask
        teach_regr = teach_pred * mask
        gt_regr = target * mask
        weights_regr = weights_soft_ind * mask[:,:,:1] * weights[...,None]
        return regr, teach_regr, gt_regr, weights_regr.squeeze(2)
        
    def get_soft_loss_weights(self, stu_pred, teach_pred, target):
        stu_pred_out =  torch.where(stu_pred>self.cls_score_thred,\
                                torch.full_like(stu_pred,1), torch.full_like(stu_pred,0))
        teach_pred_out = torch.where(teach_pred>self.cls_score_thred,\
                                torch.full_like(teach_pred,1), torch.full_like(teach_pred,0))
        target_pos_t = torch.where(target>self.cls_score_thred,\
                                torch.full_like(target,1), torch.full_like(target,0))
        # pos_normalizer
        target_pos = torch.where(target==1,\
                                torch.full_like(target,1), torch.full_like(target,0))
        if self.cls_target_ignore:
            stu_pred_res =  1 - torch.all(stu_pred_out == target_pos, dim=3).float()
            positives_t_tp_tn =  torch.all(teach_pred_out == target_pos, dim=3).float()
        else:
            # student false positive or negetive
            stu_pred_res =  1 - torch.all(stu_pred_out == target_pos_t, dim=3).float()
            # teacher true p or n
            positives_t_tp_tn =  torch.all(teach_pred_out == target_pos_t, dim=3).float()

        weights_sf = stu_pred_res / torch.clamp(stu_pred_res.sum((1,2), keepdim=True), min=1.0)

        if self.cls_use_teacher_t_only:
            weights_sf = weights_sf * positives_t_tp_tn
        self.soft_loss_weights['weights_sf'] = weights_sf

        target_pos = target_pos.sum(3)
        pos_normalizer = target_pos/torch.clamp(target_pos.sum((1,2), keepdim=True), min=1.0)
        self.soft_loss_weights['weights_gt'] = pos_normalizer


    # for distillation
    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def get_forward_ret_dict(self): #elodie
        return self.forward_ret_dict