from .detector3d_template import Detector3DTemplate


class SECONDTEACHERNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks(build_sub_branch_net=True) # elodie
        self.backbone_cfg = model_cfg['BACKBONE_3D']
        self.sub_loss_weight = model_cfg['DENSE_HEAD']['LOSS_CONFIG'].get('SUB_LOSS_WEIGHT', 1)

    def forward(self, batch_dict, is_teacher=False, teacher_ret_dict=None, teacher_data_dict=None, is_sub_model=False, batch_dict_sub=None):
        if is_sub_model:
            for cur_module in self.module_list[:2]:
                batch_dict = cur_module(batch_dict)    
            return batch_dict
            
        if batch_dict_sub is not None:
            # with torch.no_grad():
            #     for cur_module in self.module_list[:2]:
            #         batch_dict_sub = cur_module(batch_dict_sub)
            sub_multi_scale_3d_features = {}
            for feature_ in self.backbone_cfg['SUB_FEATURE_LIST']:
                sub_multi_scale_3d_features[feature_] = batch_dict_sub[feature_]
            batch_dict['sub_branch'] = sub_multi_scale_3d_features

        # print("is_teacher:",is_teacher,"   'sub_multi_scale_3d_features' in batch:", ('sub_multi_scale_3d_features' in batch_dict), "  batch_dict_sub is None:",(batch_dict_sub is None))
        for cur_module in self.module_list[:-2]:
            batch_dict = cur_module(batch_dict)

        if 'sub_branch' in batch_dict:
            batch_dict['sub_branch']['batch_size'] = batch_dict['batch_size']
            batch_dict['sub_branch']['gt_boxes'] = batch_dict['gt_boxes']

            for cur_module in self.module_list[-2:]:
                batch_dict['sub_branch'] = cur_module(batch_dict['sub_branch'])

        if is_teacher:
            forword_result = self.get_forword_result()
            return forword_result, batch_dict
            
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(teacher_ret_dict=teacher_ret_dict, student_data_dict=batch_dict, teacher_data_dict=teacher_data_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            cls_recall, cls_precision = self.dense_head.get_cls_pr_dict()
            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            # print("\nori:",self.dense_head.forward_ret_dict)

            # print("\nsub:",self.sub_dense_head.forward_ret_dict)
            # cls_recall, cls_precision = self.sub_dense_head.get_cls_pr_dict()
            # pred_dicts, recall_dicts = self.post_processing(batch_dict['sub_branch'])
            cls_dict = {
                'cls_recall':cls_recall,
                'cls_precision':cls_precision,
            }
            recall_dicts.update(cls_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, teacher_ret_dict=None, student_data_dict=None, teacher_data_dict=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss() #models/dense_heads/anchor_head_template.py

        sub_loss_rpn, sub_tb_dict = self.sub_dense_head.get_loss() #models/dense_heads/anchor_head_template.py
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'sub_branch/loss_rpn': sub_loss_rpn.item(),
            'mimic/sub_cls_preds_student_precision': sub_tb_dict['mimic/cls_preds_student_precision'],
            'mimic/sub_cls_preds_student_recall': sub_tb_dict['mimic/cls_preds_student_recall'],
            'sub_branch/rpn_hard_loss_cls': sub_tb_dict['rpn_hard_loss_cls'],
            'sub_branch/rpn_hard_loss_reg': sub_tb_dict['rpn_hard_loss_reg'],
            'sub_branch/rpn_loss_cls': sub_tb_dict['rpn_loss_cls'],
            'sub_branch/rpn_loss_loc': sub_tb_dict['rpn_loss_loc'],
            'sub_branch/rpn_loss_dir': sub_tb_dict['rpn_loss_dir'],
            **tb_dict
        }
        loss = loss_rpn + sub_loss_rpn*self.sub_loss_weight
        #disp_dict是空的?
        return loss, tb_dict, disp_dict
