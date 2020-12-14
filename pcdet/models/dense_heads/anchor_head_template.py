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

        # elodie add background class and use softmax activation
        # class 0:background 1:car 2:ped 3:cyclist
        self.add_bg_class = self.model_cfg.get('ADD_BACKGROUND_CLASS', False)
        if self.add_bg_class:
            self.num_class += 1

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)


        self.model_cfg.SOFT_LOSS_CONFIG = self.model_cfg.get('SOFT_LOSS_CONFIG', None) # elodie soft loss
        self.cls_score_thred = self.model_cfg.LOSS_CONFIG.get('CLS_SCORE_THRED', cls_score_thred)
        
        if self.model_cfg.SOFT_LOSS_CONFIG is not None:
            self.build_soft_losses(self.model_cfg.SOFT_LOSS_CONFIG)
            self.soft_loss_weights = {}
            self.mimic_cls_classes_use_only = False
        else:
            self.cls_soft_loss_type = None
            self.reg_soft_loss_type = None
            self.dir_soft_loss_type = None
            self.hint_soft_loss_type = None
            self.mimic_cls_classes_use_only = False
            
        
        self.pr_dict={
            "cls_tp_num":torch.from_numpy(np.ones(self.num_class)),
            "cls_fp_num":torch.from_numpy(np.ones(self.num_class)),
            "cls_fn_num":torch.from_numpy(np.ones(self.num_class)),
        }

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
        if self.add_bg_class:
            cls_loss_name = 'SoftmaxFocalClassificationLoss1' if losses_cfg.get('CLS_LOSS_TYPE', None) is None \
            else losses_cfg.CLS_LOSS_TYPE
            cls_alpha = losses_cfg.get('CLS_LOSS_ALPHA', 0.25)
            self.add_module(
                'cls_loss_func',
                getattr(loss_utils, cls_loss_name)(alpha=cls_alpha)
            )
        else:
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
            mimic_cls_temperature = soft_losses_cfg.CLS_LOSS.get('TEMPERATURE', 1.0)
            if self.cls_soft_loss_type in ['SigmoidFocalClassificationLoss', 'SigmoidFocalLoss']:
                self.add_module(
                    'soft_cls_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
                    )
            elif self.cls_soft_loss_type in ['WeightedKLDivergenceLoss', 'WeightedKLDivergenceLoss_v2', 'SigmoidKLDivergenceLoss','SoftmaxKLDivergenceLoss']:
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
                    'soft_cls_loss_func',
                     getattr(loss_utils, self.cls_soft_loss_type)(weighted=weighted, T=mimic_cls_temperature)
                    )
            else:
                self.add_module(
                    'soft_cls_loss_func',
                    getattr(loss_utils, self.cls_soft_loss_type)()
                )
            self.cls_soft_loss_beta = soft_losses_cfg.CLS_LOSS.get('BETA', 0.5)
            self.cls_soft_loss_modify = soft_losses_cfg.CLS_LOSS.get('MODIFY', None)
            self.cls_soft_loss_source = soft_losses_cfg.CLS_LOSS.get('SOURCE', None)
            self.cls_use_teacher_t_only = soft_losses_cfg.CLS_LOSS.get('ONLY_USE_TRUE_RET', False)
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
            self.reg_soft_loss_source = soft_losses_cfg.REG_LOSS.get('SOURCE', None)
            self.reg_soft_loss_source_weights = soft_losses_cfg.REG_LOSS.get('SOURCE_WEIGHTS', None)
            if self.reg_soft_loss_source_weights is None and self.reg_soft_loss_source is not None:
                self.reg_soft_loss_source_weights = np.ones(len(self.reg_soft_loss_source))
            self.reg_soft_loss_use_sin = soft_losses_cfg.REG_LOSS.get('USE_SIN', False)

        if self.hint_soft_loss_type is not None:
            hint_soft_loss_temperature = soft_losses_cfg.HINT_LOSS.get('TEMPERATURE', 1)

            self.add_module(
                'soft_hint_loss_func',
                getattr(loss_utils, self.hint_soft_loss_type)(T=hint_soft_loss_temperature)
            )
            self.hint_soft_loss_gamma = soft_losses_cfg.HINT_LOSS.get('GAMMA', 0.5)
            self.hint_feature_list = soft_losses_cfg.HINT_LOSS.get('FEATURE_LIST', None)
            self.hint_gt_only = soft_losses_cfg.HINT_LOSS.get('GT_ONLY', False)
            self.random_select_bg = soft_losses_cfg.HINT_LOSS.get('RANDOM_SELECT_BG', False)
            self.hint_soft_loss_source = soft_losses_cfg.HINT_LOSS.get('SOURCE', None)
            self.hint_soft_loss_source_weights = soft_losses_cfg.HINT_LOSS.get('SOURCE_WEIGHTS', None)
            if self.hint_soft_loss_source_weights is None and self.hint_soft_loss_source is not None:
                self.hint_soft_loss_source_weights = np.ones(len(self.hint_soft_loss_source))

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

    def get_cls_pr_dict(self):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        cared = box_cls_labels >= 0  # [N, num_anchors]
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)

        cls_preds = self.forward_ret_dict['cls_preds']
        cls_preds = cls_preds.view(cls_preds.shape[0], -1, self.num_class)
        if self.add_bg_class:
            cls_preds_softmax = torch.softmax(cls_preds,dim=-1)
            cls_preds_max, cls_preds_maxarg = cls_preds_softmax.max(dim=-1)
        else:
            cls_preds = torch.sigmoid(cls_preds)
            cls_preds_hot_wo_bg =  torch.where(cls_preds>self.cls_score_thred,\
                                torch.full_like(cls_preds,1), torch.full_like(cls_preds,0))
            cls_preds_hot_wo_bg_maxarg = cls_preds_hot_wo_bg.argmax(dim=-1)+1
            cls_preds_one_hot_wo_bg_sum = cls_preds_hot_wo_bg.sum(dim=-1)
            cls_preds_maxarg = torch.where(cls_preds_one_hot_wo_bg_sum>0, cls_preds_hot_wo_bg_maxarg, torch.full_like(cls_preds_hot_wo_bg_maxarg,0))
        
        # True Positive>0 / True BackGround=0 / False Negtive or False Positive=-1
        cls_preds_ret = torch.where(cls_preds_maxarg==cls_targets.long(), cls_preds_maxarg, torch.full_like(cls_preds_maxarg,-1, dtype=cls_preds_maxarg.dtype))

        for i in range(len(self.class_names)):
            # For Class i+1: True Positive>0 /False Positive=-1 / other=0
            positives_preds = torch.where(cls_preds_maxarg==(i+1), cls_preds_ret, torch.full_like(cls_preds_ret,0, dtype=cls_preds_ret.dtype))
            preds_tp = positives_preds>0
            self.pr_dict['cls_tp_num'][i] += preds_tp.float().sum()
            preds_fp = positives_preds<0
            self.pr_dict['cls_fp_num'][i] += preds_fp.float().sum()
            # For Class i+1: True Positive>0 /False Negtive=-1 / other=0
            positives_gt = torch.where(cls_targets==(i+1), cls_preds_ret, torch.full_like(cls_preds_ret,0, dtype=cls_preds_ret.dtype))
            preds_fn = positives_gt<0
            self.pr_dict['cls_fn_num'][i] += preds_fn.float().sum()

        recall = self.pr_dict['cls_tp_num']/(self.pr_dict['cls_tp_num']+self.pr_dict['cls_fn_num']) * 100
        precision = self.pr_dict['cls_tp_num']/(self.pr_dict['cls_tp_num']+self.pr_dict['cls_fp_num']) * 100
        return recall, precision

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

        if self.add_bg_class:
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class, dtype=cls_preds.dtype, device=cls_targets.device
            )
        else:
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )

        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        if self.mimic_cls_classes_use_only: # elodie
            cls_preds_per_location = cls_preds[..., self.class_index]
            cls_preds_per_location = cls_preds_per_location.view(batch_size, -1, self.class_index.shape[0])

        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        if not self.add_bg_class:
            one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        # -----------------------------Student Classification Result
        # Student False Positive:  True Positive >0, False Positive <0 elodie
        if self.add_bg_class:
            cls_preds_student_activated = torch.softmax(cls_preds,dim=-1)
            cls_preds_student_max, cls_preds_student_maxarg = cls_preds_student_activated.max(dim=-1)
        else:
            cls_preds_student_activated = torch.sigmoid(cls_preds)
            cls_preds_student_one_hot_wo_bg =  torch.where(cls_preds_student_activated>self.cls_score_thred,\
                                torch.full_like(cls_preds_student_activated,1), torch.full_like(cls_preds_student_activated,0))
            cls_preds_student_maxarg = cls_preds_student_one_hot_wo_bg.argmax(dim=-1)+1
            cls_preds_student_one_hot_wo_bg_sum = cls_preds_student_one_hot_wo_bg.sum(dim=-1)
            cls_preds_student_maxarg = torch.where(cls_preds_student_one_hot_wo_bg_sum>0, cls_preds_student_maxarg, torch.full_like(cls_preds_student_maxarg,0))
        cls_preds_student_ret = torch.where(cls_preds_student_maxarg==cls_targets.long(), cls_preds_student_maxarg, torch.full_like(cls_preds_student_maxarg,-1, dtype=cls_preds_student_maxarg.dtype))
            
        positives_student_preds = cls_preds_student_maxarg > 0
        positives_student_preds_num = positives_student_preds.float().sum(-1, keepdim=True)
        positives_s_tp = cls_preds_student_ret > 0
        positives_s_tp_num = positives_s_tp.float().sum(-1, keepdim=True)
        cls_preds_student_recall = (torch.clamp(positives_s_tp_num, min=1.0) / torch.clamp(pos_normalizer, min=1.0)).mean() 
        cls_preds_student_precision = (torch.clamp(positives_s_tp_num, min=1.0) / torch.clamp(positives_student_preds_num, min=1.0)).mean()
        # -------------------------------------------------------------
        
        tb_dict_soft = {
            'rpn_hard_loss_cls': copy.deepcopy(cls_loss.item()),
            'mimic/cls_preds_student_precision': cls_preds_student_precision.item(),
            'mimic/cls_preds_student_recall': cls_preds_student_recall.item(),
        }
        
        if teacher_result is not None and self.cls_soft_loss_type is not None: # elodie teacher
            self.soft_loss_weights['weights_gt'] = reg_weights
            
            cls_preds_teacher = teacher_result['cls_preds']
            if self.mimic_cls_classes_use_only: # elodie
                cls_preds_teacher_per_location = cls_preds_teacher[..., self.class_index]
                cls_preds_teacher_per_location = cls_preds_teacher_per_location.view(batch_size, -1, self.class_index.shape[0])
            cls_preds_teacher = cls_preds_teacher.view(batch_size, -1, self.num_class)
            
            if self.add_bg_class:
                cls_preds_teacher_activated = torch.softmax(cls_preds_teacher,dim=-1)
                cls_preds_teacher_max, cls_preds_teacher_maxarg = cls_preds_teacher_activated.max(dim=-1)
            else:
                cls_preds_teacher_activated = torch.sigmoid(cls_preds_teacher)
                cls_preds_teacher_one_hot_wo_bg =  torch.where(cls_preds_teacher_activated>self.cls_score_thred,\
                                torch.full_like(cls_preds_teacher_activated,1), torch.full_like(cls_preds_teacher_activated,0))
                cls_preds_teacher_max = cls_preds_teacher_activated.max(dim=-1)
                cls_preds_teacher_max = cls_preds_teacher_max.values
                cls_preds_teacher_maxarg = cls_preds_teacher_one_hot_wo_bg.argmax(dim=-1) + 1
                cls_preds_teacher_one_hot_wo_bg_sum = cls_preds_teacher_one_hot_wo_bg.sum(dim=-1)
                cls_preds_teacher_maxarg = torch.where(cls_preds_teacher_one_hot_wo_bg_sum>0, cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg,0))
                
            positives_teacher_preds = cls_preds_teacher_maxarg > 0 # Teacher Positive
            negatives_teacher_preds = cls_preds_teacher_maxarg == 0
            positives_teacher_preds_num = positives_teacher_preds.sum(1, keepdim=True).float()

            # Teacher Preds Result: True Positive >0, True Negtive=0, False Positive <0
            cls_preds_teacher_ret = torch.where(cls_preds_teacher_maxarg==cls_targets.long(), \
                    cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg, -1, dtype=cls_preds_teacher_maxarg.dtype))
            cls_preds_teacher_ret = torch.where(cls_preds_teacher_maxarg==0, \
                    cls_preds_teacher_maxarg, cls_preds_teacher_ret)

            # Teacher Preds False Positive with high confidence >0.8 including true positive > thred
            cls_preds_teacher_whc_ret = torch.where(cls_preds_teacher_max>0.8,\
                             cls_preds_teacher_maxarg, torch.full_like(cls_preds_teacher_maxarg,0))
            cls_preds_teacher_whc_ret = torch.where(cls_preds_teacher_whc_ret>0, cls_preds_teacher_whc_ret, cls_preds_teacher_ret)
            positives_t_whc = cls_preds_teacher_whc_ret > 0
            weights_twhc = positives_t_whc.float() / torch.clamp(positives_t_whc.float().sum(-1, keepdim=True), min=1.0)
            self.soft_loss_weights['weights_twhc'] = weights_twhc

            # teacher preds true positive num
            positives_t_tp = cls_preds_teacher_ret > 0 # Teacher Preds TP
            positives_t_tp_num = positives_t_tp.float().sum(-1, keepdim=True)
            positives_t_tp_tn = cls_preds_teacher_ret >= 0 # Teacher Preds TP&TN
            positives_t_tp_tn = positives_t_tp_tn.float()

            weights_ttp = positives_t_tp.float() / torch.clamp(positives_t_tp.float().sum(-1, keepdim=True), min=1.0)
            self.soft_loss_weights['weights_ttp'] = weights_ttp

            # Teacher cls accuracy
            positives_t_tp_tn_num = positives_t_tp_tn.sum(-1, keepdim=True)
            # cls_preds_teacher_acc = (torch.clamp(positives_t_tp_tn_num, min=1.0) / cls_preds_teacher_ret.shape[1]).mean()
            cls_preds_teacher_recall = (torch.clamp(positives_t_tp_num, min=1.0) / torch.clamp(pos_normalizer, min=1.0)).mean()
            cls_preds_teacher_precision = (torch.clamp(positives_t_tp_num, min=1.0) / torch.clamp(positives_teacher_preds_num, min=1.0)).mean()

            # print("cls_preds_teacher_acc: %.4f"%cls_preds_teacher_acc)
            # print("cls_preds_teacher_precision: %.4f"%cls_preds_teacher_precision)
            # print("cls_preds_teacher_recall: %.4f"%cls_preds_teacher_recall)
            
            # Student False Poitive
            positives_s_f = (cls_preds_student_ret < 0).float()
            positives_s_fn = torch.where(cls_targets>0, positives_s_f, torch.full_like(positives_s_f,0)) # including classify error and false negtive
            positives_s_fp = torch.where(cls_targets==0, positives_s_f, torch.full_like(positives_s_f,0))

            weights_sf = positives_s_f.float() / torch.clamp(positives_s_f.float().sum(-1, keepdim=True), min=1.0)

            weights_sfp = positives_s_fp.float() / torch.clamp(positives_s_fp.float().sum(-1, keepdim=True), min=1.0)
            weights_sfn = positives_s_fn.float() / torch.clamp(positives_s_fn.float().sum(-1, keepdim=True), min=1.0)

            if self.cls_use_teacher_t_only:
                weights_sfp = weights_sfp * positives_t_tp_tn
                weights_sfn = weights_sfn * positives_t_tp_tn
                weights_sf = weights_sf * positives_t_tp_tn
            self.soft_loss_weights['weights_sfp'] = weights_sfp
            self.soft_loss_weights['weights_sfn'] = weights_sfn
            self.soft_loss_weights['weights_sf'] = weights_sf

            ## Student cls accuracy
            # positives_s_tp_tn = cls_preds_student_ret >= 0
            # positives_s_tp_tn_num = positives_s_tp_tn.float().sum(-1, keepdim=True)
            # cls_preds_student_acc = (torch.clamp(positives_s_tp_tn_num, min=1.0) / cls_preds_student_ret.shape[1]).mean()
            # print("cls_preds_student_acc: %.4f"%cls_preds_student_acc)
            # print("cls_preds_student_precision: %.4f"%cls_preds_student_precision)
            # print("cls_preds_student_recall: %.4f"%cls_preds_student_recall)
            # print()

            if self.cls_soft_loss_type in ['SigmoidFocalClassificationLoss', 'SigmoidFocalLoss' ]:
                negative_cls_weights_t = negatives_teacher_preds * 1.0
                cls_weights_t = (negative_cls_weights_t + 1.0 * positives_teacher_preds).float()
                cls_weights_t /= torch.clamp(positives_teacher_preds_num, min=1.0)

                if self.cls_soft_loss_type == 'SigmoidFocalClassificationLoss':
                    cls_preds_teacher_one_hot = torch.zeros(
                    *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
                    )
                    cls_preds_teacher_one_hot.scatter_(-1, cls_preds_teacher_maxarg.unsqueeze(dim=-1).long(), 1.0)
                    cls_preds_teacher_one_hot = cls_preds_teacher_one_hot[..., 1:]

                    cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher_one_hot.float(), weights=cls_weights_t)  # [N, M]
                elif self.cls_soft_loss_type == 'SigmoidFocalLoss':
                    cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher_activated, weights=cls_weights_t)  # [N, M]

            else:
                if self.cls_soft_loss_source is None:
                    weights = reg_weights
                else:
                    weights = torch.full_like(positives_teacher_preds, 0, dtype=cls_weights.dtype)
                    for src, src_weights in zip(self.cls_soft_loss_source, self.cls_soft_loss_source_weights):
                        if src == "Teacher_TP":
                            weights += src_weights*weights_ttp
                        if src == "Teacher_HC":
                            weights += src_weights*weights_twhc
                        if src == "Student_FP":
                            weights += src_weights*weights_sfp
                        if src == "Student_FN":
                            weights += src_weights*weights_sfn
                        if src == "Student_F":
                            weights += src_weights*weights_sf
                        if src == "GroundTruth":
                            weights += src_weights*reg_weights
                if self.cls_soft_loss_type in ['WeightedKLDivergenceLoss', 'WeightedKLDivergenceLoss_v2', 'SigmoidKLDivergenceLoss','SoftmaxKLDivergenceLoss']:
                    if self.mimic_cls_classes_use_only: # elodie
                        weights = weights.view(batch_size, -1, self.class_index.shape[0])
                        cls_soft_loss = self.soft_cls_loss_func(cls_preds_per_location, cls_preds_teacher_per_location, weights=weights)

                        # ----------------For Test Print
                        # weights2 = weights.view(batch_size, -1, self.class_index.shape[0])
                        # cls_soft_loss1 = self.soft_cls_loss_func(cls_preds_per_location, cls_preds_teacher_per_location, weights=weights2)
                        # cls_preds_per_location_sigmoid = torch.sigmoid(cls_preds_per_location)
                        # cls_preds_teacher_per_location_sigmoid = torch.sigmoid(cls_preds_teacher_per_location)
                        # cls_preds_per_location_softmax = torch.softmax(cls_preds_per_location,dim=-1)
                        # cls_preds_teacher_per_location_softmax = torch.softmax(cls_preds_teacher_per_location,dim=-1)
                        # # cls_soft_loss = cls_soft_loss1.sum(dim=-1) * weights2
                        # cls_soft_loss = (cls_soft_loss1* weights2).sum(dim=-1) 

                        # cls_preds_sigmoid = torch.sigmoid(cls_preds)
                        # cls_soft_loss1_sigmoid = self.soft_cls_loss_func_sigmoid(cls_preds_per_location, cls_preds_teacher_per_location, weights=weights2)
                        # cls_soft_loss_sigmoid = (cls_soft_loss1_sigmoid* weights2).sum(dim=-1) 
                        # cls_soft_loss1_softmax = self.soft_cls_loss_func_softmax(cls_preds_per_location, cls_preds_teacher_per_location, weights=weights2)
                        # cls_soft_loss_softmax = (cls_soft_loss1_softmax* weights2).sum(dim=-1) 

                        # for j in range(cls_preds_per_location.shape[1]):
                        #   print("=====================")
                        #   print(j, "- cls_preds_per_location:\t", cls_preds_per_location[0,j])
                        #   print(j, "- cls_preds_teacher_per_location:\t", cls_preds_teacher_per_location[0,j])
                        #   print(j, "- cls_preds_per_location_sigmoid:\t", cls_preds_per_location_sigmoid[0,j])
                        #   print(j, "- cls_preds_teacher_per_location_sigmoid:\t", cls_preds_teacher_per_location_sigmoid[0,j])
                        #   print(j, "- kl loss sigmoid:\t", cls_soft_loss1_sigmoid[0,j])
                        #   print(j, "- kl loss sigmoid sum:\t", cls_soft_loss_sigmoid[0,j])

                        #   print(j, "- cls_preds_per_location_softmax:\t", cls_preds_per_location_softmax[0,j])
                        #   print(j, "- cls_preds_teacher_per_location_softmax:\t", cls_preds_teacher_per_location_softmax[0,j])
                        #   print(j, "- kl loss softmax:\t", cls_soft_loss1[0,j])
                        #   print(j, "- kl loss softmax2:\t", cls_soft_loss1_softmax[0,j])
                        #   print(j, "- kl loss softmax sum:\t", cls_soft_loss[0,j])

                        #   print(j, "- weights:\t", weights2[0,j])
                        #   print(j, "- ",self.cls_soft_loss_type, " loss:\t", cls_soft_loss[0,j],'\n')
                        #   print("=====================")
                        #   for n in range(6):
                        #     i = j*6+n
                        #     print(i, '- ', n, "- cls_targets:\t", cls_targets[0,i])
                        #     print(i, "- one_hot_targets:\t", one_hot_targets[0,i],'\n')
                        #     print(i, "- cls_preds_student:\t", cls_preds[0,i])
                        #     print(i, "- cls_preds_student_activated:\t", cls_preds_sigmoid[0,i])
                        #     print(i, "- cls_preds_student_maxarg:\t", cls_preds_student_maxarg[0,i])
                        #     print(i, "- cls_preds_student result:\t", cls_preds_student_ret[0,i],"\n")
                        #     print(i, "- cls_preds_teacher:\t", cls_preds_teacher[0,i])
                        #     print(i, "- cls_preds_teacher_activated:\t", cls_preds_teacher_activated[0,i])
                        #     print(i, "- cls_preds_teacher_maxarg:\t", cls_preds_teacher_maxarg[0,i])
                        #     print(i, "- cls_preds_teacher result:\t", cls_preds_teacher_ret[0,i])
                        #     print(i, "- cls_preds_teacher P include high conf:\t", cls_preds_teacher_whc_ret[0,i],'\n')
                        #     print(i, "- cls cls loss  :\t", cls_weights[0,i])
                        #     print(i, "- cls reg loss  :\t", reg_weights[0,i])
                        #     print(i, "- cls soft loss GroundTruth weights:\t", reg_weights[0,i],'\n')
                        #     print(i, "- cls soft loss Teacher_TP weights:\t", weights_ttp[0,i])
                        #     print(i, "- cls soft loss Teacher_High Conf weights:\t", weights_twhc[0,i])
                        #     print(i, "- cls soft loss Student_FP weights:\t", weights_sfp[0,i],'\n')
                        #     print(i, "- cls soft loss Student_FN weights:\t", weights_sfn[0,i],'\n')
                        #     print(i, "- cls soft loss weights:\t", weights[0,i])
                        #     print("-----------")
                    else:
                        cls_soft_loss = self.soft_cls_loss_func(cls_preds, cls_preds_teacher, weights=weights)
                else:
                    cls_soft_loss = self.soft_cls_loss_func(cls_preds_student_activated, cls_preds_teacher_activated, weights=weights)
            
            # print("\n\n---------------------start------------:\n")
            # print("\n\npositives:",positives.sum(1, keepdim=True).float())
            # print("\nnegatives:",negatives.sum(1, keepdim=True).float(),"\n")
            # print("\n\npositives_teacher:",positives_teacher_preds_num)
            # print("\nnegatives_teacher:",negatives_teacher_preds.sum(1, keepdim=True).float(),"\n")

            # for i in range(cls_targets.shape[-1]):
            # # for i in range(10):
            #     print(i, "- cls_targets:\t", cls_targets[0,i])
            #     print(i, "- one_hot_targets:\t", one_hot_targets[0,i],'\n')
            #     print(i, "- cls_preds_student:\t", cls_preds[0,i])
            #     print(i, "- cls_preds_student_activated:\t", cls_preds_student_activated[0,i])
            #     print(i, "- cls_preds_student_maxarg:\t", cls_preds_student_maxarg[0,i])
            #     print(i, "- cls_preds_student result:\t", cls_preds_student_ret[0,i],"\n")
            #     print(i, "- cls_preds_teacher:\t", cls_preds_teacher[0,i])
            #     print(i, "- cls_preds_teacher_activated:\t", cls_preds_teacher_activated[0,i])
            #     print(i, "- cls_preds_teacher_maxarg:\t", cls_preds_teacher_maxarg[0,i])
            #     print(i, "- cls_preds_teacher result:\t", cls_preds_teacher_ret[0,i])
            #     print(i, "- cls_preds_teacher P include high conf:\t", cls_preds_teacher_whc_ret[0,i],'\n')
            #     print(i, "- cls cls loss  :\t", cls_weights[0,i])
            #     print(i, "- cls reg loss  :\t", reg_weights[0,i])
            #     print(i, "- cls soft loss GroundTruth weights:\t", reg_weights[0,i],'\n')
            #     print(i, "- cls soft loss Teacher_TP weights:\t", weights_ttp[0,i])
            #     print(i, "- cls soft loss Teacher_High Conf weights:\t", weights_twhc[0,i])
            #     print(i, "- cls soft loss Student_FP weights:\t", weights_sfp[0,i],'\n')
            #     print(i, "- cls soft loss Student_FN weights:\t", weights_sfn[0,i],'\n')
            #     print(i, "- cls soft loss weights:\t", weights[0,i])
            #     print(i, "- ",self.cls_soft_loss_type, " loss:\t", cls_soft_loss[0,i],'\n')
            #     print("-----------")
            cls_soft_loss = self.cls_soft_loss_beta * cls_soft_loss.sum() / batch_size
            
            tb_dict_soft['rpn_soft_loss_cls'] = cls_soft_loss.item()
            tb_dict_soft['mimic/cls_preds_teacher_precision'] = cls_preds_teacher_precision.item()
            tb_dict_soft['mimic/cls_preds_teacher_recall'] = cls_preds_teacher_recall.item()

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

        tb_dict_soft = {
            'rpn_hard_loss_reg': copy.deepcopy(loc_loss.item()),
        }
        if teacher_result is not None and self.reg_soft_loss_type is not None: # elodie teacher
            if self.reg_soft_loss_source is not None:
                weights = torch.full_like(positives, 0, dtype=reg_weights.dtype)
                for src, src_loss_weights in zip(self.reg_soft_loss_source, self.reg_soft_loss_source_weights):
                    if src == "Teacher_TP":
                        weights += src_loss_weights*self.soft_loss_weights['weights_ttp']
                    if src == "Teacher_HC":
                        weights += src_loss_weights*self.soft_loss_weights['weights_twhc']
                    if src == "Student_FP":
                        weights += src_loss_weights*self.soft_loss_weights['weights_sfp']
                    if src == "Student_FN":
                        weights += src_weights*self.soft_loss_weights['weights_sfn']
                    if src == "Student_F":
                        weights += src_weights*self.soft_loss_weights['weights_sf']
                    if src == "GroundTruth":
                        weights += src_loss_weights*self.soft_loss_weights['weights_gt']
            else:
                weights = None

            if self.reg_soft_loss_type == 'BoundedRegressionLoss':
                box_preds_teacher = teacher_result['box_preds']
                box_dir_cls_preds_teacher = teacher_result.get('dir_cls_preds', None)
                box_preds_teacher = box_preds_teacher.view(batch_size, -1,
                                        box_preds_teacher.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                        box_preds_teacher.shape[-1])
                if self.reg_soft_loss_use_sin:
                    box_preds_teacher_sin, reg_targets_teacher_sin = self.add_sin_difference(box_preds_teacher, box_reg_targets)
                    loc_soft_loss_src = self.soft_reg_loss_func(box_preds_sin, box_preds_teacher_sin, reg_targets_sin, target_teacher=reg_targets_teacher_sin, weights=weights) 
                else:
                    loc_soft_loss_src = self.soft_reg_loss_func(box_preds, box_preds_teacher, box_reg_targets, weights=weights) 
                
                loc_soft_loss = self.reg_soft_loss_alpha *loc_soft_loss_src.sum() / batch_size
                # print("reg loss:\n\tloc_loss before:",loc_loss,"\n\tloc_soft_loss:",loc_soft_loss)

                tb_dict_soft['rpn_soft_loss_reg'] = loc_soft_loss.item()

                if self.reg_soft_loss_modify is not None:
                    loc_loss = (1-self.reg_soft_loss_modify)*loc_loss + self.reg_soft_loss_modify * loc_soft_loss
                else:
                    loc_loss = loc_loss + loc_soft_loss
                

                # print("\n\tloc_loss after:",loc_loss)

            # for i in range(box_preds.shape[1]):
            # # for i in range(10):
            #     if weights[0,i]>0:
            #         index = i%6
            #         index2 = int(i/6)
            #         print(index2,',',index, "- box_reg_targets:\t", box_reg_targets[0,i])
            #         try:
            #             print(index2,',',index, "- reg_targets_teacher_sin:\t", reg_targets_teacher_sin[0,i])
            #             print(index2,',',index, "- box_preds_teacher_sin:\t", box_preds_teacher_sin[0,i])
            #         except:
            #             pass
            #         print(index2,',',index, "- box_preds_teacher:\t", box_preds_teacher[0,i])
            #         print(index2,',',index, "- box_preds_student:\t", box_preds[0,i])
            #         print(index2,',',index, "- box_preds_student_sin:\t", box_preds_sin[0,i])
            #         print(index2,',',index, "-", self.reg_soft_loss_type," loc_soft_loss:\t", loc_soft_loss_src[0,i])
            #         print(index2,',',index, "- weights:\t", weights[0,i])
            #         print(index2,',',index, "- soft_loc_loss:\t", loc_soft_loss_src[0,i],"\n")
            #         print(index2,',',index, "- hard_loc_loss:\t", loc_loss_src[0,i],"\n")
            #         print("-----------")
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
        
        if tb_dict_soft is not None:
            tb_dict.update(tb_dict_soft)
        return box_loss, tb_dict

    def draw_features(self, feature_map_t, feature_map_s, id):
        feature_map_t = feature_map_t[0]
        im_t = np.squeeze(feature_map_t.detach().cpu().numpy())
        im_t = np.transpose(im_t, [1, 2, 0])

        feature_map_s = feature_map_s[0]
        im_s = np.squeeze(feature_map_s.detach().cpu().numpy())
        im_s = np.transpose(im_s, [1, 2, 0])

        # plt.figure(figsize=(50,24))
        channel_num = 15
        fig,axs=plt.subplots(2,channel_num,figsize=(channel_num*3,16),constrained_layout=True)
        for i in range(channel_num):
            cmap = 'jet'
            ax = axs[0][i]
            # ax = plt.subplot(2, 3, i+1, figsize=(10,10))
            if i == channel_num-1:
                ax.imshow(np.mean(im_t,axis=-1), cmap=plt.get_cmap(cmap))
                print("im_t_mean:",np.mean(im_t,axis=-1))
                ax.set_title("teacher_mean")

            else:
                ax.imshow(im_t[50:150, :50, i], cmap=plt.get_cmap(cmap))
                print("im_t[50:150, :50, i]:",im_t[50:150, :50, i])
                ax.set_title("teacher_"+str(i),fontsize=12)

            ax = axs[1][i]
            # ax = plt.subplot(2, 3, i+4)
            if i == channel_num-1:
                ax.imshow(np.mean(im_s,axis=-1), cmap=plt.get_cmap(cmap))
                print("im_s_mean:",np.mean(im_s,axis=-1))
                print("np.mean(im_s,axis=-1):",np.mean(im_s,axis=-1).shape)
                ax.set_title("student_mean")

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
        if self.hint_soft_loss_source is None:
            weights = None
        else:
            weights = torch.full_like(self.soft_loss_weights['weights_ttp'], 0, dtype=self.soft_loss_weights['weights_ttp'].dtype)
            for src, src_loss_weights in zip(self.hint_soft_loss_source, self.hint_soft_loss_source_weights):
                if src == "Teacher_TP":
                    weights += src_loss_weights*self.soft_loss_weights['weights_ttp']
                if src == "Teacher_HC":
                    weights += src_loss_weights*self.soft_loss_weights['weights_twhc']
                if src == "Student_FP":
                    weights += src_loss_weights*self.soft_loss_weights['weights_sfp']
                if src == "Student_FN":
                    weights += src_weights*self.soft_loss_weights['weights_sfn']
                if src == "Student_F":
                    weights += src_weights*self.soft_loss_weights['weights_sf']
                if src == "GroundTruth":
                    weights += src_loss_weights*self.soft_loss_weights['weights_gt']
            
        for i, feature_ in enumerate(self.hint_feature_list):
            if feature_[:len('x_conv')] == 'x_conv' or feature_ == 'encoded_spconv_tensor':
                # print("feature_:",feature_)
                if feature_ == 'encoded_spconv_tensor':
                    student_feature = student_data_dict[feature_]
                    teacher_feature = teacher_data_dict[feature_]
                else:  
                    student_feature = student_data_dict['multi_scale_3d_features'][feature_]
                    teacher_feature = teacher_data_dict['multi_scale_3d_features'][feature_]
                if self.hint_gt_only:
                    assert 'voxel_coords_inbox_dict' in student_data_dict, 'voxel_coords_inbox_dict not in student_data_dict!'
                    student_feature_coor = student_feature.indices.long()
                    teacher_feature_coor = teacher_feature.indices.long()
                    gt_feature_coor = student_data_dict['voxel_coords_inbox_dict'][feature_]

                    student_gt_coor_unique = torch.cat((student_feature_coor,gt_feature_coor),0).unique(sorted=True,return_inverse=True,return_counts=True, dim=0)
                    # s_gt_inverse = student_gt_coor_unique[1][:student_feature_coor.shape[0]]
                    s_gt_inverse = student_gt_coor_unique[1][:student_feature_coor.shape[0]]
                    s_gt_sorted, s_gt_indices = torch.sort(s_gt_inverse)
                    s_gt_coor_indices = torch.zeros(student_gt_coor_unique[2].shape[0], dtype=torch.long).cuda()
                    s_gt_coor_indices[s_gt_sorted] = s_gt_indices
                    student_gt_mask = student_gt_coor_unique[2]==2
                    student_gt_coor_index = s_gt_coor_indices[torch.arange(0,student_gt_coor_unique[2].shape[0])[student_gt_mask]]
                    if self.random_select_bg:
                        rand_mask = torch.rand(student_feature_coor.shape[0])
                        rand_mask[student_gt_coor_index] = 0
                        mask =  torch.where(rand_mask>0.5,\
                                torch.full_like(rand_mask,1), torch.full_like(rand_mask,0))
                        student_gt_coor_bg = student_feature_coor[mask.bool()] 
                        student_gt_coor_bg = student_gt_coor_bg if student_gt_coor_bg.shape[0]<student_gt_coor_index.shape[0] else student_gt_coor_bg[:student_gt_coor_index.shape[0]]
                        student_gt_coor_index_bg = torch.arange(0,student_feature_coor.shape[0])[mask.bool()].cuda()
                        student_gt_coor_index_bg = student_gt_coor_index_bg if student_gt_coor_index_bg.shape[0]<student_gt_coor_index.shape[0] else student_gt_coor_index_bg[:student_gt_coor_index.shape[0]]
                        student_gt_coor_index = torch.cat((student_gt_coor_index,student_gt_coor_index_bg),0)
                    
                    student_gt_coor = student_feature_coor[student_gt_coor_index]
                    
                    st_gt_coor_unique = torch.cat((teacher_feature_coor,student_gt_coor),0).unique(sorted=True,return_inverse=True,return_counts=True, dim=0)# get the gt coordinates in teacher models 
                    st_gt_inverse = st_gt_coor_unique[1][:teacher_feature_coor.shape[0]]
                    st_gt_sorted, st_gt_indices = torch.sort(st_gt_inverse)
                    st_gt_coor_indices = torch.zeros(st_gt_coor_unique[2].shape[0], dtype=torch.long).cuda()
                    st_gt_coor_indices[st_gt_sorted] = st_gt_indices
                    st_gt_mask = torch.arange(0,st_gt_coor_unique[2].shape[0])[st_gt_coor_unique[2]==2]
                    st_gt_coor_index = st_gt_coor_indices[st_gt_mask]

                    sst_gt_inverse = st_gt_coor_unique[1][teacher_feature_coor.shape[0]:]
                    sst_gt_sorted, sst_gt_indices = torch.sort(sst_gt_inverse)
                    sst_gt_coor_indices = torch.zeros(st_gt_coor_unique[2].shape[0], dtype=torch.long).cuda()
                    sst_gt_coor_indices[sst_gt_sorted] = sst_gt_indices
                    sst_gt_coor_index = sst_gt_coor_indices[st_gt_mask]
                    # for i in range(sst_gt_coor_index.shape[0]):
                        # print(student_feature.indices[student_gt_coor_index][sst_gt_coor_index[i]],teacher_feature.indices[st_gt_coor_index[i]])
                    hint_loss_src = self.soft_hint_loss_func(student_feature.features[sst_gt_coor_index], teacher_feature.features[st_gt_coor_index])
                    # else:
                        # hint_loss_src = self.soft_hint_loss_func(student_feature.features[student_gt_coor_index], teacher_feature.features[st_gt_coor_index])
                else:
                    student_feature_coor = student_feature.indices.long()
                    teacher_feature_coor = teacher_feature.indices.long()
                    
                    st_gt_coor_unique = torch.cat((teacher_feature_coor,student_feature_coor),0).unique(sorted=True,return_inverse=True,return_counts=True, dim=0)# get the gt coordinates in teacher models 
                    st_gt_inverse = st_gt_coor_unique[1][:teacher_feature_coor.shape[0]]
                    st_gt_sorted, st_gt_indices = torch.sort(st_gt_inverse)
                    st_gt_coor_indices = torch.zeros(st_gt_coor_unique[2].shape[0], dtype=torch.long).cuda()
                    st_gt_coor_indices[st_gt_sorted] = st_gt_indices
                    st_gt_mask = st_gt_coor_unique[2]==2
                    st_gt_coor_index = st_gt_coor_indices[torch.arange(0,st_gt_coor_unique[2].shape[0])[st_gt_mask]]

                    # teacher_coor_index = torch.sparse_coo_tensor(teacher_feature_coor.t(), torch.arange(1,teacher_feature_coor.shape[0]+1).cuda(), torch.Size(tuple(teacher_feature_coor.max(dim=0)[0]+1)))
                    # teacher_coor_index = teacher_coor_index.to_dense()
                    # # teacher_coor_index =torch.zeros(tuple(teacher_feature_coor.max(dim=0)[0]+1),dtype=torch.long)
                    # # teacher_coor_index =teacher_coor_index.index_put_(tuple(teacher_feature_coor.t()), torch.arange(1,teacher_feature_coor.shape[0]+1))
                    # align_index = teacher_coor_index[tuple(student_feature_coor.t())]
                    # align_index -= 1
                    
                    # aligned_teacher_feature = teacher_feature.features[align_index]

                    # aligned_teacher_coor = teacher_feature_coor[align_index]

                    # print("\n\naligned_teacher_coor:",aligned_teacher_coor,"\nstudent_feature_coor:",student_feature_coor)
                    # for i in range(student_feature_coor.shape[0]):
                    #     print(i," - ",aligned_teacher_coor[i],"\n",student_feature_coor[i])
                    hint_loss_src = self.soft_hint_loss_func(student_feature.features,teacher_feature.features[st_gt_coor_index])
                hint_loss_src = hint_loss_src.sum()
                # print("hint_loss:",hint_loss)
            else:
                student_feature = student_data_dict[feature_]
                teacher_feature = teacher_data_dict[feature_]
                student_feature = student_feature.permute(0, 2, 3, 1) # [N,H,W,C]
                student_feature = student_feature.view(student_feature.shape[0], -1, student_feature.shape[-1])

                teacher_feature = teacher_feature.permute(0, 2, 3, 1) # [N,H,W,C]
                teacher_feature = teacher_feature.view(teacher_feature.shape[0], -1, teacher_feature.shape[-1])

                batch_size = int(student_feature.shape[0])
                if weights is not None:
                    weights = weights.view(batch_size, -1, self.num_anchors_per_location)
                    weights = weights.sum(dim=-1)
                    hint_loss_src = self.soft_hint_loss_func(student_feature,teacher_feature,weights=weights)
                else:
                    hint_loss_src = self.soft_hint_loss_func(student_feature,teacher_feature)
                hint_loss_src = hint_loss_src.sum()/batch_size

            # frame_id =  student_data_dict['frame_id'][0]
            # print("frame_id:",frame_id)
            # self.draw_features(teacher_feature, student_feature, frame_id)


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
