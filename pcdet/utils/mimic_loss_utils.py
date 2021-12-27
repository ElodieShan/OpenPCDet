import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedKLDivergenceLoss_v2(nn.Module):
    def __init__(self, T=1.0, weighted=True):
        super(WeightedKLDivergenceLoss_v2, self).__init__()
        self.T = T
        self.weighted = weighted
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)

        input = F.log_softmax(input/self.T, dim=-1)
        target = F.softmax(target/self.T, dim=-1)

        klloss = F.kl_div(input, target, reduction='none').sum(dim=-1) * self.T * self.T 
        if self.weighted:
            klloss = klloss* weights
        else:
            klloss = klloss.mean(dim=-1)
        return klloss


class BoundedRegressionLoss_v2(nn.Module):

    def __init__(self, margin: float = 0.001):
        """
        Args:
            alpha: Weighting parameter to balance soft and hard loss.
            margin: teacher bounded margin. 
        """
        super(BoundedRegressionLoss_v2, self).__init__()
        self.margin = margin

    def forward(self, input_student: torch.Tensor, input_teacher: torch.Tensor, target: torch.Tensor, target_teacher=None, weights=None):
        """
        Args:
            input_student/input_teacher: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.

        Returns:
            loss: (B, #anchors) float tensor.
                BoundedRegressionLoss.
        """
        target = torch.where(torch.isnan(target), input_student, target)  # ignore nan targets

        l2_st = torch.abs(input_student - input_teacher)

        l2_student = torch.pow((input_student - target), 2)-self.margin
        if target_teacher is not None:
            l2_teacher = torch.pow((input_teacher - target_teacher), 2)
        else:
            l2_teacher = torch.pow((input_teacher - target), 2)

        soft_loss = torch.where(l2_student>l2_teacher, l2_st, torch.full_like(l2_student,0))
        if weights is None:
            soft_loss = soft_loss.sum(dim=-2)
            # soft_loss = soft_loss / (l2_student>l2_teacher).sum(1, keepdim=True).float()
        else:
            # soft_loss = soft_loss.sum(dim=-1)*weights
            soft_loss = (soft_loss*weights.view(weights.shape[0],-1,1)).sum(dim=-2)
        return soft_loss
