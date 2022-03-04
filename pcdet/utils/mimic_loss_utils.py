import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def teach_weights(preds, teach_preds):
    diff = torch.sigmoid(teach_preds) - torch.sigmoid(preds)
    return torch.pow(diff, 2).sum(dim=2)

class WeightedKLDivergenceLoss_v2(nn.Module):
    def __init__(self, T=1.0, weighted=True, activated=False):
        super(WeightedKLDivergenceLoss_v2, self).__init__()
        self.T = T
        self.weighted = weighted
        self.activated = activated
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None):
        if not self.activated:
            input = torch.sigmoid(input)
            target = torch.sigmoid(target)

        input = F.log_softmax(input/self.T, dim=-1)
        target = F.softmax(target/self.T, dim=-1)

        klloss = F.kl_div(input, target, reduction='none').sum(dim=-1) * self.T * self.T 
        # print( klloss.shape, weights.shape)

        if self.weighted:
            klloss = klloss* weights
        else:
            klloss = klloss.reshape([klloss.shape[0],-1])
            klloss = klloss.mean(dim=-1)
        # print("klloss:", klloss.sum())
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
        # print("before:", soft_loss.sum())
        if weights is None:
            soft_loss = soft_loss.sum(dim=-2)
            # soft_loss = soft_loss / (l2_student>l2_teacher).sum(1, keepdim=True).float()
        else:
            # soft_loss = soft_loss.sum(dim=-1)*weights
            soft_loss = (soft_loss*weights.view(weights.shape[0],-1,1)).sum(dim=-2)

        return soft_loss

class BoundedRegressionLoss_l1(nn.Module):
    
    def __init__(self, margin: float = 0.001):
        """
        Args:
            alpha: Weighting parameter to balance soft and hard loss.
            margin: teacher bounded margin. 
        """
        super(BoundedRegressionLoss_l1, self).__init__()
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

        l2_student = torch.abs(input_student - target)-self.margin
        if target_teacher is not None:
            l2_teacher = torch.abs(input_teacher - target_teacher)
        else:
            l2_teacher = torch.abs(input_teacher - target)

        soft_loss = torch.where(l2_student>l2_teacher, l2_st, torch.full_like(l2_student,0))
        # print("before:", soft_loss.sum())
        if weights is None:
            soft_loss = soft_loss.sum(dim=-2)
            # soft_loss = soft_loss / (l2_student>l2_teacher).sum(1, keepdim=True).float()
        else:
            # soft_loss = soft_loss.sum(dim=-1)*weights
            # soft_loss = (soft_loss*weights.view(weights.shape[0],-1,1)).sum(dim=-2))
            soft_loss = (soft_loss*weights.view(weights.shape[0],-1,1)).sum(dim=(0,1),keepdim=False)
        # for i in range(10):
        #     print("----------- i:", i)
        #     print("target:", target[0,i])
        #     print("input_teacher:", input_teacher[0,i])
        #     print("input_student:", input_student[0,i])
        #     print("weights:", weights[0,i])

        # print("soft_loss:", soft_loss.sum(), soft_loss)
        return soft_loss
    
class HintL2Loss(nn.Module):
    def __init__(self, T=1.0, normalize=False):
        super(HintL2Loss, self).__init__()
        self.T = T
        self.normalize = normalize

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None):
        if self.normalize:
            # print("input before:",input, input.shape)
            input = F.normalize(input,p=2,dim=1)
            target = F.normalize(target,p=2,dim=1)
        l2_hint_loss_src = torch.pow((input - target), 2)
        # print("input:",input.shape)
        # print("l2_hint_loss_src:", l2_hint_loss_src.sum(dim=-1).shape)
        # print("l2_hint_loss_src front:", l2_hint_loss_src.sum(dim=-1)[:10])
        # print("l2_hint_loss_src bg:", l2_hint_loss_src.sum(dim=-1)[10:])
        # for i in range(10):
        #     print(i,input[i],target[i])
        # for i in range(10):
        #     print(input.shape[0]-i-1,input[input.shape[0]-i-1],target[input.shape[0]-i-1])
        if weights is None:
            l2_hint_loss = l2_hint_loss_src.sum(dim=-1)
        else:
            l2_hint_loss = l2_hint_loss_src.sum(dim=-1)*weights
            
        # for i in range(l2_hint_loss_src.shape[1]):
        # # for i in range(10):
        #     if weights is not None and weights[0,i]>0:
        #             print(i, "- student:\t", input[0,i,:20])
        #             print(i, "- teacher:\t", target[0,i,:20])
        #             print(i, "- weights:\t", weights[0,i])
        #             print(i, "- l2_hint_loss_src:\t", l2_hint_loss_src[0,i,:20])
        #             print(i, "- l2_hint_loss:\t", l2_hint_loss[0,i])

        #             print("-----------\n")

        return l2_hint_loss
    
    
class HintFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, T=1.0, normalize=False):
        super(HintFocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.iou_peak=0.0
        self.T = T
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None):
        print(weights.shape, target.shape, input.shape)
        positive_mask = torch.where(weights > self.iou_peak, torch.full_like(weights,1), torch.full_like(weights,0))
        print("sum:",positive_mask.sum())
        positive_mask = positive_mask[...,None]
        target = target * positive_mask
        
        ce_loss = torch.log(1 - input) * (1-target) + torch.log(input) * target
        pos_loss = -ce_loss * target * positive_mask
        
        neg_scale_factor = torch.pow(torch.abs(input), self.gamma) * self.alpha
        neg_loss = -torch.log(1 - input) * neg_scale_factor * (1-positive_mask)

        vfl_loss = neg_loss + pos_loss
        hint_loss = vfl_loss.sum(dim=-1)
        print("neg_loss:", neg_loss.sum(dim=-1), "\tpos_loss:", pos_loss.sum(dim=-1), "\thint_loss:", hint_loss)
        return hint_loss