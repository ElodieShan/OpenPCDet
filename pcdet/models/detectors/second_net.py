from .detector3d_template import Detector3DTemplate
import torch

class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.backbone_cfg = model_cfg['BACKBONE_3D']

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
            batch_dict['sub_multi_scale_3d_features'] = sub_multi_scale_3d_features

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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
            cls_dict = {
                'cls_recall':cls_recall,
                'cls_precision':cls_precision,
            }
            recall_dicts.update(cls_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, teacher_ret_dict=None, student_data_dict=None, teacher_data_dict=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss(teacher_ret_dict=teacher_ret_dict, student_data_dict=student_data_dict, teacher_data_dict=teacher_data_dict) #models/dense_heads/anchor_head_template.py
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        #disp_dict是空的?
        return loss, tb_dict, disp_dict
