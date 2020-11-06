import numpy as np
import torch
import torch.nn as nn
import copy

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
import matplotlib.pyplot as plt

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training, cls_score_thred=None):
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

        self.model_cfg.SOFT_LOSS_CONFIG = self.model_cfg.get('SOFT_LOSS_CONFIG', None) # elodie soft loss
        
        if self.model_cfg.SOFT_LOSS_CONFIG is not None:
            self.build_soft_losses(self.model_cfg.SOFT_LOSS_CONFIG)
            self.cls_score_thred = cls_score_thred
        else:
            self.cls_soft_loss_type = None
            self.reg_soft_loss_type = None
            self.dir_soft_loss_type = None
            self.hint_soft_loss_type = None
            

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
        self.hint_soft_loss_type = None if soft_losses_cfg.get('HINT_LOSS', None) is None \
            else soft_losses_cfg.HINT_LOSS.TYPE

        if self.cls_soft_loss_type is not None:
            if self.cls_soft_loss_type in ['SigmoidFocalClassificationLoss', 'SigmoidFocalLoss']:
                self.add_module(
                    'soft_cls_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
                    )
            elif self.cls_soft_loss_type in ['WeightedKLDivergenceLoss']:
                weighted = soft_losses_cfg.CLS_LOSS.get('WEIGHTED', True)
                self.add_module(
                    'soft_cls_loss_func',
                     getattr(loss_utils, self.cls_soft_loss_type)(weighted=weighted)
                    )
            else:
                self.add_module(
                    'soft_cls_loss_func',
                    getattr(loss_utils, self.cls_soft_loss_type)()
                )
            self.cls_soft_loss_beta = soft_losses_cfg.CLS_LOSS.get('BETA', 0.5)
            self.cls_soft_loss_modify = soft_losses_cfg.CLS_LOSS.get('MODIFY', None)
            self.cls_soft_loss_source = soft_losses_cfg.CLS_LOSS.get('SOURCE', None)
            self.cls_soft_loss_source_weights = soft_losses_cfg.CLS_LOSS.get('SOURCE_WEIGHTS', None)
            if self.cls_soft_loss_source_weights is None and self.cls_soft_loss_source is not None:
                self.cls_soft_loss_source_weights = np.ones(len(self.cls_soft_loss_source))

        if self.reg_soft_loss_type is not None:
            self.add_module(
                'soft_reg_loss_func',
                getattr(loss_utils, self.reg_soft_loss_type)(\
                    margin=soft_losses_cfg.REG_LOSS.get('MARGIN', 0.001))
            )
            self.reg_soft_loss_alpha = soft_losses_cfg.REG_LOSS.get('ALPHA', 0.5)
            self.reg_soft_loss_modify = soft_losses_cfg.REG_LOSS.get('MODIFY', None)

        if self.hint_soft_loss_type is not None:
            self.add_module(
                'soft_hint_loss_func',
                getattr(loss_utils, self.hint_soft_loss_type)()
            )
            self.hint_soft_loss_gamma = soft_losses_cfg.HINT_LOSS.get('GAMMA', 0.5)
            self.hint_feature_list = soft_losses_cfg.HINT_LOSS.get('FEATURE_LIST', None)

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

        tb_dict_soft = None
        if teacher_result is not None and self.cls_soft_loss_type is not None: # elodie teacher
            cls_preds_teacher = teacher_result['cls_preds']
            cls_preds_teacher = cls_preds_teacher.view(batch_size, -1, self.num_class)
            cls_preds_teacher_sigmoid = torch.sigmoid(cls_preds_teacher)
            cls_preds_teacher_one_hot_wo_bg =  torch.where(cls_preds_teacher_sigmoid>self.cls_score_thred,\
                             torch.full_like(cls_preds_teacher_sigmoid,1), torch.full_like(cls_preds_teacher_sigmoid,0))
            cls_preds_teacher_max = cls_preds_teacher_sigmoid.max(dim=-1)
            cls_preds_teacher_maxarg = cls_preds_teacher_one_hot_wo_bg.argmax(dim=-1) + 1
            cls_preds_teacher_one_hot_wo_bg_sum = cls_preds_teacher_one_hot_wo_bg.sum(dim=-1)
            cls_preds_teacher_maxarg = torch.where(cls_preds_teacher_one_hot_wo_bg_sum>0, cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg,0))
            
            cls_preds_teacher_one_hot = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            cls_preds_teacher_one_hot.scatter_(-1, cls_preds_teacher_maxarg.unsqueeze(dim=-1).long(), 1.0)
            cls_preds_teacher_one_hot = cls_preds_teacher_one_hot[..., 1:]

            # print("cls_preds_teacher_one_hot:",cls_preds_teacher_one_hot)

            positives_t = cls_preds_teacher_maxarg > 0 # Teacher Positive
            negatives_t = cls_preds_teacher_maxarg == 0

            negative_cls_weights_t = negatives_t * 1.0
            cls_weights_t = (negative_cls_weights_t + 1.0 * positives_t).float()

            pos_normalizer_t = positives_t.sum(1, keepdim=True).float()
            cls_weights_t /= torch.clamp(pos_normalizer_t, min=1.0)

            # Teacher True Positive >0, False Positive <0
            cls_preds_teacher_maxarg_tp = torch.where(cls_preds_teacher_maxarg==cls_targets.long(), \
                    cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg, -1, dtype=cls_preds_teacher_maxarg.dtype))
            cls_preds_teacher_maxarg_tp = torch.where(cls_preds_teacher_maxarg==0, \
                    cls_preds_teacher_maxarg, cls_preds_teacher_maxarg_tp)

            positives_t_tp = cls_preds_teacher_maxarg_tp > 0
            weights_ttp = positives_t_tp.float() / torch.clamp(positives_t_tp.float().sum(-1, keepdim=True), min=1.0)

            # Teacher Preds False Positive with high confidence >0.5 including true positive > thred
            cls_preds_teacher_maxarg_whc = torch.where(cls_preds_teacher_max.values>0.8,\
                             cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg,0))
            cls_preds_teacher_maxarg_whc = torch.where(cls_preds_teacher_maxarg_whc>0, cls_preds_teacher_maxarg_whc, cls_preds_teacher_maxarg_tp)
            positives_t_whc = cls_preds_teacher_maxarg_whc > 0
            weights_twhc = positives_t_whc.float() / torch.clamp(positives_t_whc.float().sum(-1, keepdim=True), min=1.0)

            # Student False Positive:  True Positive >0, False Positive <0
            cls_preds_student_sigmoid = torch.sigmoid(cls_preds)
            cls_preds_student_one_hot_wo_bg =  torch.where(cls_preds_student_sigmoid>self.cls_score_thred,\
                             torch.full_like(cls_preds_student_sigmoid,1), torch.full_like(cls_preds_student_sigmoid,0))
            cls_preds_student_maxarg = cls_preds_student_one_hot_wo_bg.argmax(dim=-1)+1
            cls_preds_student_one_hot_wo_bg_sum = cls_preds_student_one_hot_wo_bg.sum(dim=-1)
            cls_preds_student_maxarg = torch.where(cls_preds_student_one_hot_wo_bg_sum>0, cls_preds_student_maxarg, torch.full_like(cls_preds_student_maxarg,0))

            cls_preds_student_fp = torch.where(cls_preds_student_maxarg==cls_targets.long(), cls_preds_student_maxarg, torch.full_like(cls_preds_student_maxarg,-1, dtype=cls_preds_student_maxarg.dtype))
            positives_s_fp = cls_preds_student_fp < 0
            weights_sfp = positives_s_fp.float() / torch.clamp(positives_s_fp.float().sum(-1, keepdim=True), min=1.0)

            if self.cls_soft_loss_type in ['SigmoidFocalClassificationLoss', 'SigmoidFocalLoss' ]:
                if self.cls_soft_loss_type == 'SigmoidFocalClassificationLoss':
                    cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher_one_hot.float(), weights=cls_weights_t)  # [N, M]
                elif self.cls_soft_loss_type == 'SigmoidFocalLoss':
                    cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher_sigmoid, weights=cls_weights_t)  # [N, M]

            else:
                if self.cls_soft_loss_source is None:
                    weights = positives_t.float()
                    weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                else:
                    weights = torch.full_like(positives_t, 0, dtype=cls_weights.dtype)
                    for src, src_weights in zip(self.cls_soft_loss_source, self.cls_soft_loss_source_weights):
                        if src == "Teacher_TP":
                            weights += src_weights*weights_ttp
                        if src == "Teacher_HC":
                            weights += src_weights*weights_twhc
                        if src == "Student_FP":
                            weights += src_weights*weights_sfp
                cls_soft_loss = self.soft_cls_loss_func(cls_preds_student_sigmoid, cls_preds_teacher_sigmoid, weights=weights)
            
            # print("\n\n---------------------start------------:\n")
            # print("\n\npositives:",positives.sum(1, keepdim=True).float())
            # print("\nnegatives:",negatives.sum(1, keepdim=True).float(),"\n")
            # print("\n\npositives_teacher:",positives_t.sum(1, keepdim=True).float())
            # print("\nnegatives_teacher:",negatives_t.sum(1, keepdim=True).float(),"\n")
            # cls_preds_sigmoid = torch.sigmoid(cls_preds)

            # for i in range(cls_targets.shape[-1]):
            # # for i in range(10):
            #     print(i, "- cls_targets:\t", cls_targets[0,i])
            #     print(i, "- one_hot_targets:\t", one_hot_targets[0,i],'\n')
            #     print(i, "- cls_preds_student:\t", cls_preds[0,i])
            #     print(i, "- cls_preds_student_sigmoid:\t", cls_preds_sigmoid[0,i])
            #     print(i, "- cls_preds_student_maxarg:\t", cls_preds_student_maxarg[0,i])
            #     print(i, "- cls_preds_student FP:\t", cls_preds_student_fp[0,i],"\n")
            #     print(i, "- cls_preds_teacher:\t", cls_preds_teacher[0,i])
            #     print(i, "- cls_preds_teacher_sigmoid:\t", cls_preds_teacher_sigmoid[0,i])
            #     print(i, "- cls_preds_teacher_one_hot:\t", cls_preds_teacher_one_hot[0,i])
            #     print(i, "- cls_preds_teacher_maxarg:\t", cls_preds_teacher_maxarg[0,i])
            #     print(i, "- cls_preds_teacher TP:\t", cls_preds_teacher_maxarg_tp[0,i])
            #     print(i, "- cls_preds_teacher P include high conf:\t", cls_preds_teacher_maxarg_whc[0,i],'\n')
            #     print(i, "- cls soft loss Teacher_TP weights:\t", weights_ttp[0,i])
            #     print(i, "- cls soft loss Teacher_High Conf weights:\t", weights_twhc[0,i])
            #     print(i, "- cls soft loss Student_FP Conf weights:\t", weights_sfp[0,i],'\n')
            #     print(i, "- cls soft loss weights:\t", weights[0,i])
            #     print(i, "- ",self.cls_soft_loss_type, " loss:\t", cls_soft_loss[0,i],'\n')
            #     print("-----------")
            cls_soft_loss = self.cls_soft_loss_beta * cls_soft_loss.sum() / batch_size

            tb_dict_soft = {
                'rpn_soft_loss_cls': cls_soft_loss.item(),
                'rpn_hard_loss_cls': copy.deepcopy(cls_loss.item()),
            }
            # print("cls loss before:\n\tcls_loss before:",cls_loss,"\n\tcls_soft_losss:",cls_soft_loss)
            if self.cls_soft_loss_modify is not None:
                cls_loss = (1-self.cls_soft_loss_modify)*cls_loss + self.cls_soft_loss_modify * cls_soft_loss
            else:
                cls_loss = cls_loss + cls_soft_loss
        # print("\tcls_loss:",cls_loss)

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        if tb_dict_soft is not None:
            tb_dict.update(tb_dict_soft)

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

    def get_box_reg_layer_loss(self, teacher_result=None):
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
                loc_soft_loss = self.reg_soft_loss_alpha *loc_soft_loss.sum() / batch_size
                # print("reg loss:\n\tloc_loss before:",loc_loss,"\n\tloc_soft_loss:",loc_soft_loss)

                if self.reg_soft_loss_modify is not None:
                    loc_loss = (1-self.reg_soft_loss_modify)*loc_loss + self.reg_soft_loss_modify * loc_soft_loss
                else:
                    loc_loss = loc_loss + loc_soft_loss
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

    def draw_features(self, feature_map_t, feature_map_s, id):
        feature_map_t = feature_map_t[0]
        im_t = np.squeeze(feature_map_t.detach().cpu().numpy())
        im_t = np.transpose(im_t, [1, 2, 0])

        feature_map_s = feature_map_s[0]
        im_s = np.squeeze(feature_map_s.detach().cpu().numpy())
        im_s = np.transpose(im_s, [1, 2, 0])

        # plt.figure(figsize=(50,24))

        fig,axs=plt.subplots(2,3,figsize=(10,16),constrained_layout=True)
        for i in range(3):
            cmap = 'jet'
            ax = axs[0][i]
            # ax = plt.subplot(2, 3, i+1, figsize=(10,10))
            if i == 2:
                ax.imshow(np.mean(im_t,axis=-1), cmap=plt.get_cmap(cmap))
                print("im_t_mean:",np.mean(im_t,axis=-1))

            else:
                ax.imshow(im_t[50:150, :50, i], cmap=plt.get_cmap(cmap))
                print("im_t[50:150, :50, i]:",im_t[50:150, :50, i])
            ax.set_title("teacher_"+str(i),fontsize=12)

            ax = axs[1][i]
            # ax = plt.subplot(2, 3, i+4)
            if i == 2:
                ax.imshow(np.mean(im_s,axis=-1), cmap=plt.get_cmap(cmap))
                print("im_s_mean:",np.mean(im_s,axis=-1))
                print("np.mean(im_s,axis=-1):",np.mean(im_s,axis=-1).shape)
            else:
                ax.imshow(im_s[50:150, :50, i], cmap=plt.get_cmap(cmap))
                print("im_s[50:150, :50, i]:",im_s[50:150, :50, i])

            ax.set_title("student_"+str(i),fontsize=12)

        
        savefile = "/home/elodie/temp/" + str(id) + '.png'
        plt.savefig(savefile, dpi=100)
        plt.clf()
        # plt.close()

    def get_hint_loss(self, student_data_dict=None, teacher_data_dict=None):
        hint_loss = 0.0
        for feature_ in self.hint_feature_list:
            student_feature = student_data_dict[feature_]
            teacher_feature = teacher_data_dict[feature_]
            
            frame_id =  student_data_dict['frame_id'][0]
            print("frame_id:",frame_id)
            self.draw_features(teacher_feature, student_feature, frame_id)

            hint_loss_src = self.soft_hint_loss_func(student_feature,teacher_feature)
            batch_size = int(student_feature.shape[0])
            hint_loss_src = hint_loss_src.sum()/batch_size
            hint_loss = hint_loss + self.hint_soft_loss_gamma * hint_loss_src
        
        tb_dict = {
            'hint_loss': hint_loss.item()
        }
        return hint_loss, tb_dict

    def get_loss(self, teacher_ret_dict=None, student_data_dict=None, teacher_data_dict=None):
        cls_loss, tb_dict = self.get_cls_layer_loss(teacher_result=teacher_ret_dict)
        box_loss, tb_dict_box = self.get_box_reg_layer_loss(teacher_result=teacher_ret_dict)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss 
        
        if self.hint_soft_loss_type is not None:
            hint_loss, tb_dict_hint = self.get_hint_loss(student_data_dict=student_data_dict, teacher_data_dict=teacher_data_dict)
            # print("hint_loss:",hint_loss)
            tb_dict.update(tb_dict_hint)
            rpn_loss = rpn_loss + hint_loss

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
