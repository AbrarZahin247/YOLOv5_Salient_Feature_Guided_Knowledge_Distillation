import torch
import torch.nn as nn

from utils.ult_loss import ComputeLoss
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

class AlignStudentFeatureToTeacher(nn.Module):
    def __init__(self, input_dim, output_dim, device=None):
        super(AlignStudentFeatureToTeacher, self).__init__()
        # Linear layer to convert input_dim (19200) to output_dim (25200)
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Move the model to the specified device
        if device:
            self.to(device)
    
    def forward(self, x):
        # Apply the linear layer
        x = self.linear(x)
        return x


class NetwithLoss(torch.nn.Module):
    def __init__(self, teacher, student, device):
        super().__init__()
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        self.sl1 = nn.SmoothL1Loss()
        # self.mse=nn.MSELoss()
        
        self.dynamic_student = nn.Sequential(
            # Flatten the dimensions 1, 2, and 3 (3, 80, 80) into a single dimension
            nn.Flatten(start_dim=1, end_dim=3)  # Flatten dimensions 1, 2, 3
        )

    @torch.cuda.amp.autocast()
    def forward(self, imgs, backgrounds, targets,image_size=640):
        
        
        # dummy_image = torch.zeros((1, 3, image_size, image_size), device=self.device)
        # targets = torch.Tensor([[0, 0, 0, 0, 0, 0]]).to(self.device)
        # features= self.student(dummy_image, target=targets)  # forward
        # teacher_feature = self.teacher(dummy_image, target=targets)
        # print(features[0].shape)
        # print(teacher_feature[0].shape)
        # _,_, student_channel, student_out_size, _ = features[0].shape
        # _,_, teacher_channel, teacher_out_size, _ = teacher_feature[0].shape
        # stu_feature_adapt = nn.Sequential(nn.Conv2d(student_channel, teacher_channel, 3, padding=1, stride=int(student_out_size / teacher_out_size)), nn.ReLU()).to(self.device)
        
        
        
        
        # Move inputs to the correct device
        imgs = imgs.to(self.device)
        backgrounds = backgrounds.to(self.device)
        targets = targets.to(self.device)

        pred = self.student(imgs)
        # pred=stu_feature_adapt(pred)
        # print(f"student ==> {pred[0].shape}")
        pred_back = self.student(backgrounds)
        diff_pred = pred[0] - pred_back[0]
        diff_pred=self.dynamic_student(diff_pred)
        
        # batch, anchor, dim_w, dim_h, class_and_bbox = diff_pred.shape
        
        predT_Raw = self.teacher(imgs)
        # print(f"teacher ==> {predT_Raw[0].shape}")
        predT_Raw_back = self.teacher(backgrounds)
        diff_predT = predT_Raw[0] - predT_Raw_back[0]
        
        
        converter = nn.Linear(diff_pred.shape[1], diff_predT.shape[1]).to(self.device)
        # AlignStudentFeatureToTeacher(,device=self.device)
        student_like = converter(diff_pred.permute(0, 2, 1))  # Permute to shape [24, 12, 19200]
        diff_pred = student_like.permute(0, 2, 1)
        
        
        
        
        hint_loss = self.sl1(diff_pred, diff_predT)
        
        # hint_loss_factor = 1e-5
        hint_loss_factor = 100
        hint_loss = hint_loss * hint_loss_factor
        # print(f"hint loss ==> {hint_loss}")
        # Loss
        compute_loss = ComputeLoss(self.student)
        loss, loss_items = compute_loss(pred, targets)  # scaled by batch_size

        return loss + hint_loss, loss_items