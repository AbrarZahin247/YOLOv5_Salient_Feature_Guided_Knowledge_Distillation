import torch
import torch.nn as nn

from utils.ult_loss import ComputeLoss
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


class NetwithLoss(torch.nn.Module):
    def __init__(self, teacher, student, device):
        super().__init__()
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        self.sl1 = nn.SmoothL1Loss()

    @torch.cuda.amp.autocast()
    def forward(self, imgs, backgrounds, targets):
        # Move inputs to the correct device
        imgs = imgs.to(self.device)
        backgrounds = backgrounds.to(self.device)
        targets = targets.to(self.device)

        pred = self.student(imgs)
        pred_back = self.student(backgrounds)
        diff_pred = pred[0] - pred_back[0]
        batch, anchor, dim_w, dim_h, class_and_bbox = diff_pred.shape
        
        predT_Raw = self.teacher(imgs)
        predT_Raw_back = self.teacher(backgrounds)
        diff_predT = predT_Raw[0] - predT_Raw_back[0]
        diff_predT = diff_predT.reshape(batch, anchor, dim_w, dim_h, class_and_bbox)

        first_anchor_feature, second_anchor_feature, third_anchor_feature = (
            diff_pred[:, 0, :, :, :], 
            diff_pred[:, 1, :, :, :], 
            diff_pred[:, 2, :, :, :]
        )
        first_anchor_feature_T, second_anchor_feature_T, third_anchor_feature_T = (
            diff_predT[:, 0, :, :, :], 
            diff_predT[:, 1, :, :, :], 
            diff_predT[:, 2, :, :, :]
        )

        hint_loss = (
            self.sl1(first_anchor_feature, first_anchor_feature_T) + 
            self.sl1(second_anchor_feature, second_anchor_feature_T) + 
            self.sl1(third_anchor_feature, third_anchor_feature_T)
        )
        
        hint_loss_factor = 1e-4
        hint_loss = hint_loss * hint_loss_factor
        
        # Loss
        compute_loss = ComputeLoss(self.student)
        loss, loss_items = compute_loss(pred, targets)  # scaled by batch_size

        return loss + hint_loss, loss_items