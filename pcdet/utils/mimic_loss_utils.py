import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class HintMimic(nn.Module):

    def __init__(self, margin: float = 0.0, alpha: float = 1.0):
        """
        Args:
            margin: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for hard and easy loss.
        """
        super(HintMimic, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, student_loss_input: torch.Tensor, teacher_loss_input: torch.Tensor):
        """
        Args:
            student_input: (B, #anchors, #classes or codes) float tensor.
                Predicted logits for each class
            teacher_input: (B, #anchors, #classes or codes) float tensor.

        Returns:
            soft_weighted_loss: sum/batch_size.
        """
        teacher_loss_input = teacher_loss_input + self.margin 
        student_loss_soft = student_loss_input[student_loss_input>teacher_loss_input]
        student_loss_soft = self.alpha * student_loss_soft.sum()

        return student_loss_soft

