from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, is_teacher=False, teacher_ret_dict=None):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(teacher_ret_dict=teacher_ret_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        elif is_teacher:
            forword_result = self.get_forword_result()
            return forword_result
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_forword_result(self):
        return self.dense_head.get_forward_ret_dict()

    def get_training_loss(self,teacher_ret_dict=None):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss(teacher_ret_dict=teacher_ret_dict) #models/dense_heads/anchor_head_template.py
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        #disp_dict是空的?
        return loss, tb_dict, disp_dict
