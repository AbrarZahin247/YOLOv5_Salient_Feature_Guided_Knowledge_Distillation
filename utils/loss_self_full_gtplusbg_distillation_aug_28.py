import torch
import torch.nn as nn

from utils.ult_loss import ComputeLoss
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


class SelfDistLoss(torch.nn.Module):
    def __init__(self, student, device):
        super().__init__()
        self.student = student.to(device)
        self.device = device

        self.sl1 = nn.SmoothL1Loss()
        self.mse=nn.MSELoss()
        
        # self.dynamic_student = nn.Sequential(
        #     # Flatten the dimensions 1, 2, and 3 (3, 80, 80) into a single dimension
        #     nn.Flatten(start_dim=1, end_dim=3)  # Flatten dimensions 1, 2, 3
        # )

    @torch.cuda.amp.autocast()
    def forward(self, imgs, targets,gt_images,bg_images):
        # Move inputs to the correct device
        imgs = imgs.to(self.device)
        gt_images = gt_images.to(self.device)
        targets = targets.to(self.device)

        pred = self.student(imgs)
        pred_gt=self.student(gt_images)
        pred_bg=self.student(bg_images)
        complementary_combined=pred_gt[0]+pred_bg[0]
        
        dist_factor=1e-2
        complementary_loss=self.mse(pred[0],complementary_combined)*dist_factor
        # print(f"complementary loss : {complementary_loss}")
        # print(f"target shape {targets.shape}---- pred shape {pred[0].shape} ---- pred_gt shape {pred_gt[0].shape}")
        # Loss
        compute_loss = ComputeLoss(self.student)
        loss, loss_items = compute_loss(pred, targets)  # scaled by batch_size
        # loss_gt, loss_items = compute_loss(pred_gt, targets)  # scaled by batch_size
        # print(f"loss_gt {loss_gt} loss {loss}")
        return loss+complementary_loss, loss_items