import torch
import torch.nn as nn
import cv2
import numpy as np

from utils.ult_loss import ComputeLoss
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def show_all_images_from_batch(images, delay=1):
    """
    Displays all images from a batched tensor using OpenCV with a specified delay between each.

    Parameters:
    images (torch.Tensor): A tensor of shape [batch_size, channels, height, width] representing the batch of images.
    delay (int): Delay in seconds between displaying each image.

    Example usage:
    show_all_images_from_batch(images, delay=1)
    """
    # Loop through each image in the batch
    for i in range(images.size(0)):
        # Extract the i-th image from the batch
        image = images[i]  # Shape is [3, 640, 640] assuming input shape [60, 3, 640, 640]

        # Convert from PyTorch tensor to numpy array and transpose to HWC format
        image = image.permute(1, 2, 0).cpu().numpy()  # Shape is now [640, 640, 3]

        # Convert the image from range [0, 1] to [0, 255] if needed
        # Assuming image tensor values are in range [0, 1]
        image = (image * 255).astype(np.uint8)

        # Display the image using cv2
        cv2.imshow(f'Image {i+1}', image)

        # Wait for the specified delay (converted to milliseconds)
        cv2.waitKey(delay * 1000)

        # Destroy the window to prepare for the next image
        cv2.destroyWindow(f'Image {i+1}')

    # Optionally, destroy all windows at the end
    cv2.destroyAllWindows()

def transform_tensor(student_tensor, teacher_tensor):
    """
    Transforms a tensor to a target size by truncating excess elements or padding with zeros.

    Args:
        student_tensor (torch.Tensor): Student tensor of shape [batch_size, anchors, gridx, gridy, features].
        teacher_tensor (torch.Tensor): Teacher tensor of shape [batch_size, t_intermediate, features].

    Returns:
        torch.Tensor: Transformed student tensor of shape [batch_size, t_intermediate, features], or None if no transformation is needed.
    """
    # Reshape teacher tensor
    t_batch_size, t_intermediate, t_features = teacher_tensor.shape
    teacher_3d = teacher_tensor.view(t_batch_size, -1, t_features)
    
    # Reshape student tensor
    batch_size, anchors, gridx, gridy, features = student_tensor.shape
    student_intermediate_feat = anchors * gridx * gridy
    student_3d = student_tensor.view(batch_size, -1, features)
    
    # If teacher tensor requires more elements than student tensor has
    if t_intermediate > student_intermediate_feat:
        # Allocate a tensor of zeros for padding the student tensor
        student_like_teacher_zero_tensor = torch.zeros(
            t_batch_size, t_intermediate, t_features,
            device=student_tensor.device, dtype=student_tensor.dtype
        )
        student_like_teacher_zero_tensor[:, :student_intermediate_feat, :] = student_3d
        return student_like_teacher_zero_tensor, teacher_3d
    
    # If no transformation is required, return None
    return None

class SalientDistillLoss(nn.Module):
    def __init__(self, student, teacher, device):
        super().__init__()
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.device = device
        self.sl1 = nn.SmoothL1Loss()

    @torch.cuda.amp.autocast()  # For automatic mixed precision (reduce memory usage)
    def forward(self, imgs, targets, gt_masks):
        loss_factor = 1e-2
        inv_masks = 1 - gt_masks
        
        # Process background images by applying inverted masks
        bg_images = imgs * inv_masks
        imgs, bg_images, targets = imgs.to(self.device), bg_images.to(self.device), targets.to(self.device)

        # Forward pass for student and teacher models
        pred_student = self.student(imgs)
        pred_student_bg = self.student(bg_images)
        pred_teacher = self.teacher(imgs)
        pred_teacher_bg = self.teacher(bg_images)
        
        # Calculate salient features for student and teacher
        salient_feature_student = pred_student[0] - pred_student_bg[0]
        salient_feature_teacher = pred_teacher[0] - pred_teacher_bg[0]
        
        # Adjust the shape of the student tensor to match the teacher tensor
        transformed_tensors = transform_tensor(salient_feature_student, salient_feature_teacher)
        if transformed_tensors is not None:
            salient_feature_student, salient_feature_teacher = transformed_tensors

        # Compute the loss
        compute_loss = ComputeLoss(self.student)
        loss, loss_items = compute_loss(pred_student, targets)  # scaled by batch_size
        loss_smoothL1 = self.sl1(salient_feature_teacher, salient_feature_student)  # scaled by batch_size
        
        # Combine losses
        return loss + loss_smoothL1, loss_items