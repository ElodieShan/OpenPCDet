from .detector3d_template import Detector3DTemplate


class PartA2Net(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


    def forward(self, batch_dict, is_teacher=False, teacher_ret_dict=None, teacher_data_dict=None, batch_dict_sub=None):
        if is_teacher:
            for cur_module in self.module_list[:6]:
                batch_dict = cur_module(batch_dict)
            forword_result = self.get_forword_result()
            return forword_result, batch_dict

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(teacher_ret_dict=teacher_ret_dict, student_data_dict=batch_dict, teacher_data_dict=teacher_data_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_training_loss(self, teacher_ret_dict=None, student_data_dict=None, teacher_data_dict=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss(teacher_ret_dict=teacher_ret_dict, student_data_dict=student_data_dict, teacher_data_dict=teacher_data_dict) #models/dense_heads/anchor_head_template.py
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict