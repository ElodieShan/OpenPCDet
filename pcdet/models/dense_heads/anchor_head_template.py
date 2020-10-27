import numpy as np
import torch
import torch.nn as nn
import copy

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.build_soft_losses(self.model_cfg.SOFT_LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def build_soft_losses(self, soft_losses_cfg):
        self.cls_soft_loss_type = None if soft_losses_cfg.get('CLS_LOSS', None) is None \
            else soft_losses_cfg.CLS_LOSS.TYPE
        self.reg_soft_loss_type = None if soft_losses_cfg.get('REG_LOSS', None) is None \
            else soft_losses_cfg.REG_LOSS.TYPE
        self.dir_soft_loss_type = None if soft_losses_cfg.get('DIR_LOSS', None) is None \
            else soft_losses_cfg.DIR_LOSS.TYPE

        if self.cls_soft_loss_type is not None:
            if self.cls_soft_loss_type == 'SigmoidFocalClassificationLoss':
                self.add_module(
                    'soft_cls_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
                    )
            else:
                self.add_module(
                    'soft_cls_loss_func',
                    getattr(loss_utils, self.cls_soft_loss_type)()
                )
            self.cls_soft_loss_beta = soft_losses_cfg.CLS_LOSS.get('BETA', 0.5)

        if self.reg_soft_loss_type is not None:
            self.add_module(
                'soft_reg_loss_func',
                getattr(loss_utils, self.reg_soft_loss_type)(\
                    alpha=soft_losses_cfg.REG_LOSS.get('ALPHA', 0.5),\
                    margin=soft_losses_cfg.REG_LOSS.get('MARGIN', 0.001))
            )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self, teacher_result=None):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        batch_size = int(cls_preds.shape[0])

        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        if teacher_result is not None and self.cls_soft_loss_type is not None: # elodie teacher
            cls_preds_teacher = teacher_result['cls_preds']
            cls_preds_teacher = cls_preds_teacher.view(batch_size, -1, self.num_class)
            if self.cls_soft_loss_type == 'SigmoidFocalClassificationLoss':
                cls_preds_teacher_max = cls_preds_teacher.argmax(dim=-1)
                cls_preds_teacher_one_hot = torch.zeros(
                    *list(cls_preds_teacher_max.shape), self.num_class, dtype=cls_preds_teacher_max.dtype, device=cls_preds_teacher_max.device
                )
                cls_preds_teacher_one_hot.scatter_(-1, cls_preds_teacher_max.unsqueeze(dim=-1).long(), 1.0)
                cls_preds_teacher_one_hot = torch.where(cls_preds_teacher>0, cls_preds_teacher_one_hot, torch.full_like(cls_preds_teacher_one_hot,0))
                
                cls_preds_teacher_max = cls_preds_teacher_one_hot.argmax(dim=-1)
                positives_t = cls_preds_teacher_max > 0
                negatives_t = cls_preds_teacher_max == 0
                negative_cls_weights_t = negatives_t * 1.0
                cls_weights_t = (negative_cls_weights_t + 1.0 * positives_t).float()

                pos_normalizer_t = positives_t.sum(1, keepdim=True).float()
                cls_weights_t /= torch.clamp(pos_normalizer_t, min=1.0)
                cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher_one_hot.float(), weights=cls_weights_t/2.0)  # [N, M]
            else:
                weights = positives.float()
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher, weights=weights)

            cls_soft_loss = cls_soft_loss.sum() / batch_size
            # print("cls loss:\n\tcls_loss before:",cls_loss,"\n\tcls_soft_losss:",cls_soft_loss)
            cls_loss = cls_loss + self.cls_soft_loss_beta * cls_soft_loss
            # print("\tcls_loss after:",cls_loss)


        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self, teacher_result=None, beta=0.5):
        # teacher_result is forward_ret_dict of teacher model
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        loc_loss = loc_loss_src.sum() / batch_size

        if teacher_result is not None and self.reg_soft_loss_type is not None: # elodie teacher
            if self.reg_soft_loss_type == 'BoundedRegressionLoss':
                box_preds_teacher = teacher_result['box_preds']
                box_dir_cls_preds_teacher = teacher_result.get('dir_cls_preds', None)
                box_preds_teacher = box_preds_teacher.view(batch_size, -1,
                                        box_preds_teacher.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                        box_preds_teacher.shape[-1])

                loc_soft_loss = self.soft_reg_loss_func(box_preds, box_preds_teacher, box_reg_targets) 
                loc_soft_loss = loc_soft_loss.sum() / batch_size
                # print("reg loss:\n\tloc_loss before:",loc_loss,"\n\tloc_soft_loss:",loc_soft_loss)
                loc_loss = loc_loss + beta * loc_soft_loss
                # print("\tloc_loss after:",loc_loss)

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss_src = self.dir_loss_func(dir_logits, dir_targets, weights=weights)

            dir_loss = dir_loss_src.sum() / batch_size

            if teacher_result is not None and self.dir_soft_loss_type is not None: # elodie teacher
                print("teacher_dir")
                box_dir_cls_preds_teacher = teacher_result.get('dir_cls_preds', None)
                dir_logits_teacher = box_dir_cls_preds_teacher.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                dir_loss_teacher = self.dir_loss_func(dir_logits_teacher, dir_targets, weights=weights)

                dir_loss_student = dir_loss_src
                dir_loss_student_soft = dir_loss_student[dir_loss_student>dir_loss_teacher]
                dir_loss_student_soft = dir_loss_student_soft.sum() / batch_size
                dir_loss = dir_loss + alpha*dir_loss_student_soft

            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self, teacher_ret_dict=None):
        cls_loss, tb_dict = self.get_cls_layer_loss(teacher_result=teacher_ret_dict)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss(teacher_result=teacher_ret_dict)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss 
        
        tb_dict['rpn_loss'] = rpn_loss.item()
        # tb_dict= {
        #     rpn_loss_cls: ,
        #     rpn_loss_loc: ,
        #     rpn_loss_dir: ,
        # }
        return rpn_loss, tb_dict

    def get_forward_ret_dict(self): #elodie
        return self.forward_ret_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
