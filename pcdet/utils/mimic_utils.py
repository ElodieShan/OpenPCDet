import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils

def get_same_indices(high_res_indices, low_res_indices, return_same_indices_low=True):
    """
        Input should be long type
    """
    combined_indices_unique = torch.cat((high_res_indices,low_res_indices),0).unique(sorted=True,return_inverse=True,return_counts=True, dim=0) 
    combined_indices_inverse = combined_indices_unique[1][:high_res_indices.shape[0]]
    indices_sorted, sorted_index = torch.sort(combined_indices_inverse)
    combined_indices = torch.zeros(combined_indices_unique[2].shape[0], dtype=torch.long).cuda()
    combined_indices[indices_sorted] = sorted_index
    combined_indices_mask = combined_indices_unique[2]==2
    same_indices_high = combined_indices[torch.arange(0,combined_indices_unique[2].shape[0])[combined_indices_mask]]
    diff_indices_mask = torch.ones(high_res_indices.shape[0]).bool()
    diff_indices_mask[same_indices_high] = False
    diff_indices = torch.arange(0,high_res_indices.shape[0])[diff_indices_mask].cuda()
    # diff_indices = combined_indices[torch.arange(0,combined_indices_unique[2].shape[0])[~combined_indices_mask]]
    if return_same_indices_low:
        low_inverse = combined_indices_unique[1][high_res_indices.shape[0]:]
        low_sorted, low_sorted_indices = torch.sort(low_inverse)
        low_indices = torch.zeros(combined_indices_unique[2].shape[0], dtype=torch.long).cuda()
        low_indices[low_sorted] = low_sorted_indices
        same_indices_low = low_indices[torch.arange(0,combined_indices_unique[2].shape[0])[combined_indices_mask]]
        return same_indices_high, same_indices_low, diff_indices
    else:
        return same_indices_high, None, diff_indices
