import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class SoftWeightedLoss(nn.Module):

    def __init__(self, margin: float = 0.0, alpha: float = 1.0):
        """
        Args:
            margin: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for hard and easy loss.
        """
        super(SoftWeightedLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, student_input: torch.Tensor, teacher_input: torch.Tensor, target: torch.Tensor, loss_func):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

