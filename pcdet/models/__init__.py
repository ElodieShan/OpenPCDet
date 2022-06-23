from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset, device):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, device=device
    )
    return model


def load_data(batch_dict, device):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().to(device).contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().to(device)
        else:
            batch_dict[key] = torch.from_numpy(val).float().to(device)
