from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict, batch_dict_teacher=None, model_teacher=None, model_copy=None, batch_dict_sub=None): #elodie
        import copy

        load_data_to_gpu(batch_dict)

        if model_teacher is not None: # elodie
            load_data_to_gpu(batch_dict_teacher)
            if batch_dict_sub is not None:
                load_data_to_gpu(batch_dict_sub)
            else:
                batch_dict_sub=None
            with torch.no_grad():
                teacher_ret_dict, teacher_data_dict = model_teacher(batch_dict_teacher, is_teacher=True, batch_dict_sub=batch_dict_sub)
                
            ret_dict, tb_dict, disp_dict = model(batch_dict, teacher_ret_dict=teacher_ret_dict, teacher_data_dict=teacher_data_dict)
        else:
            if model_copy is not None: # elodie
                load_data_to_gpu(batch_dict_sub)
                with torch.no_grad():
                    sub_data_dict = model_copy(batch_dict_sub, is_sub_model=True)
                ret_dict, tb_dict, disp_dict = model(batch_dict, batch_dict_sub=sub_data_dict)
            else:
                if batch_dict_sub is not None:
                    load_data_to_gpu(batch_dict_sub)
                    ret_dict, tb_dict, disp_dict = model(batch_dict, batch_dict_sub=batch_dict_sub)
                else:
                    ret_dict, tb_dict, disp_dict = model(batch_dict)


        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
