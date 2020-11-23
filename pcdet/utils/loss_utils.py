import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        
        # formula is same as :
        # loss = target * -log(sigmoid(input)) + (1 - target) * -log(1 - sigmoid(input))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
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
        # print("pred_sigmoid:",pred_sigmoid)
        # print("target:",target)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        # bce_loss2 = F.binary_cross_entropy_with_logits(input, target, reduce=False) # combine sigmoid and BCE
        # pt2 = torch.exp(-bce_loss2)
        # focal_weight2 = alpha_weight * (1-pt2)**self.gamma
        # F_loss = focal_weight2 * bce_loss2

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss

class MSE(nn.Module):
    '''
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    '''
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor ):
        loss = F.mse_loss(input, target, reduction='none').sum(dim=-1)
        # print("loss:",loss)
        # print("loss.shape",loss.shape)
        # loss = loss*weights
        # for i in range(loss.shape[-1]):
        #     if loss[0,i]>0:
        #         print(loss[0,i])
        return loss*weights

class HintL2Loss(nn.Module):
    def __init__(self, T=1.0):
        super(HintL2Loss, self).__init__()
        self.T = T

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None):
        input = input.permute(0, 2, 3, 1) # [N,H,W,C]
        input = input.view(input.shape[0], -1, input.shape[-1])

        target = target.permute(0, 2, 3, 1) # [N,H,W,C]
        target = target.view(target.shape[0], -1, target.shape[-1])

        l2_hint_loss_src = torch.pow((input - target), 2)

        if weights is None:
            l2_hint_loss = l2_hint_loss_src.mean(dim=-1)
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

        # print("\nl2_hint_loss sum:",l2_hint_loss.sum(),"\n\n")
        return l2_hint_loss


class HintKLDivergenceLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T=1.0, weighted=True):
        super(HintKLDivergenceLoss, self).__init__()
        self.T = T
        self.weighted = weighted

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        # loss = F.kl_div(F.log_softmax(input/self.T, dim=1),
                        # F.softmax(target/self.T, dim=1),
                        # reduction='batchmean') * self.T * self.T
        # print("input",input)
        # input = input.view(input.shape[0], input.shape[1], -1)
        # target = target.view(target.shape[0], target.shape[1], -1)
        # loss = F.kl_div(F.log_softmax(input/self.T, dim=-1),
        #                 F.softmax(target/self.T, dim=-1),
        #                 reduction='none') * self.T * self.T 
        # print("loss:",loss)
        # loss = loss.permute(0, 2, 1).sum(dim=-1)
        input = input.permute(0, 2, 3, 1) # [N,H,W,C]
        input = input.view(input.shape[0], -1, input.shape[-1])

        target = target.permute(0, 2, 3, 1) # [N,H,W,C]
        target = target.view(target.shape[0], -1, target.shape[-1])
        loss = F.kl_div(F.log_softmax(input/self.T, dim=-1),
                        F.softmax(target/self.T, dim=-1),
                        reduction='none') * self.T * self.T 
        loss = loss.sum(dim=-1)

        if self.weighted:
            loss = loss * weights
        else:
            loss = loss.mean(dim=-1)
        # print("loss:",loss.sum()/2)

        return loss

class WeightedKLDivergenceLoss(nn.Module):
    def __init__(self, T=1.0, weighted=True):
        super(WeightedKLDivergenceLoss, self).__init__()
        self.T = T
        self.weighted = weighted
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        input = F.log_softmax(input/self.T, dim=-1)
        target = F.softmax(target/self.T, dim=-1)

        klloss = F.kl_div(input, target, reduction='none').sum(dim=-1) * self.T * self.T 
        if self.weighted:
            klloss = klloss* weights
        else:
            klloss = klloss.mean(dim=-1)
        # print("klloss:",klloss.sum()/2)
        return klloss

class SigmoidKLDivergenceLoss(nn.Module):
    def __init__(self, T=1.0, weighted=True):
        super(SigmoidKLDivergenceLoss, self).__init__()
        self.T = T
        self.weighted = weighted

    @staticmethod
    def klloss_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ 
        KL Loss = P(x)*log(P(x)/Q(x)) =P(x)*(log(P(x))-log(Q(x))
        ------------------------------>P(x)*|log(P(x))-log(Q(x)|

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        input_logsig = F.logsigmoid(input)
        target_sig = F.sigmoid(target)
        target_logsig = F.logsigmoid(target)

        loss = target_sig * torch.abs(target_logsig-input_logsig)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        klloss = self.klloss_with_logits(input/self.T,target/self.T).sum(dim=-1) * self.T * self.T 
        if self.weighted:
            klloss = klloss* weights
        else:
            klloss = klloss.mean(dim=-1)
        return klloss

class BoundedRegressionLoss(nn.Module):

    def __init__(self, margin: float = 0.001):
        """
        Args:
            alpha: Weighting parameter to balance soft and hard loss.
            margin: teacher bounded margin. 
        """
        super(BoundedRegressionLoss, self).__init__()
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

        l2_student = torch.pow((input_student - target), 2)-self.margin
        if target_teacher is not None:
            l2_teacher = torch.pow((input_teacher - target_teacher), 2)
        else:
            l2_teacher = torch.pow((input_teacher - target), 2)

        soft_loss = torch.where(l2_student>l2_teacher, l2_student, torch.full_like(l2_student,0))
        if weights is None:
            soft_loss = soft_loss / (l2_student>l2_teacher).sum(1, keepdim=True).float()
        else:
            soft_loss = soft_loss.sum(dim=-1)*weights
        # soft_loss = 
        # soft_loss = soft_loss.mean(dim=1)
        # print("soft_loss:",soft_loss)
        # for i in range(1000):
        #     if soft_loss2[0,i] > 0:
        #         print(soft_loss2[0,i])
        return soft_loss



class MMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD, self).__init__()
        """
        kernel_mul: kernel bandwith
        kernel_num: kernel_num
        fix_sigma: fix std
        """
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target):
        """ kernel mat
        source: batch * sample_size_1 * feature_size
        target: batch * sample_size_2 * feature_size

            return: [(sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
                            [	K_ss K_st
                                K_ts K_tt ]]
        """
        batch_size = int(source.size()[0])
        n_samples = int(source.size()[1]) + int(target.size()[1])

        total = torch.cat([source, target], dim=1)  # 合并在一起

        total0 = total.unsqueeze(1).expand(int(total.size(0)), \
                                        int(total.size(1)), \
                                        int(total.size(1)), \
                                        int(total.size(2)))
        total1 = total.unsqueeze(2).expand(int(total.size(0)), \
                                        int(total.size(1)), \
                                        int(total.size(1)), \
                                        int(total.size(2)))
        L2_distance = torch.pow((total0 - total1), 2).sum(-1)  # kernel |x-y|

        # kernel bandwidth
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data,dim=-1).sum(dim=-1) / (n_samples ** 2 - n_samples)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        self.kernel_num_ = torch.tensor(np.arange(self.kernel_num))
        self.kernel_mul_ = torch.tensor(np.ones(self.kernel_num)*self.kernel_mul).long()
        bandwidth_list = torch.matmul(bandwidth.view(-1,1), torch.pow(self.kernel_mul_,self.kernel_num_).double().view(1,-1))

        # exp(-|x-y|/bandwith)
        for i in range(batch_size):
            kernel_val_batch = [torch.exp(-L2_distance[i] / bandwidth_temp) for \
                        bandwidth_temp in bandwidth_list[i]]
            if i == 0:
                kernel_val = sum(kernel_val_batch).unsqueeze(0)
            else:
                kernel_val = torch.cat((kernel_val, sum(kernel_val_batch).unsqueeze(0)))

        return kernel_val

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        batch_size = int(input.size()[0])
        input = input.view(batch_size,-1,input.shape[-1])
        target = target.view(batch_size,-1,target.shape[-1])
        n_samples = int(input.size()[1])

        kernels = self.guassian_kernel(input, target)

        XX = kernels[:, :n_samples, :n_samples]  # input<->input
        YY = kernels[:, n_samples:, n_samples:]  # Target<->Target
        XY = kernels[:, :n_samples, n_samples:]  # input<->Target
        YX = kernels[:, n_samples:, :n_samples]  # Target<->input
        loss_src = XX + YY - XY - YX
        loss = loss_src.view(batch_size,-1).mean(dim=-1)
        return loss



def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)

