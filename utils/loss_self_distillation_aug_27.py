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


class SelfDistLoss(torch.nn.Module):
    def __init__(self, student, device):
        super().__init__()
        self.student = student.to(device)
        self.device = device

        self.sl1 = nn.SmoothL1Loss()
        # self.mse=nn.MSELoss()
        
        # self.dynamic_student = nn.Sequential(
        #     # Flatten the dimensions 1, 2, and 3 (3, 80, 80) into a single dimension
        #     nn.Flatten(start_dim=1, end_dim=3)  # Flatten dimensions 1, 2, 3
        # )

    @torch.cuda.amp.autocast()
    def forward(self, imgs, targets,gt_images):
        # print(torch.unique(gt_images))
        # show_all_images_from_batch(gt_images)
        # print(gt_images.shape)
        # Move inputs to the correct device
        
        
        imgs = imgs.to(self.device)
        gt_images = gt_images.to(self.device)
        targets = targets.to(self.device)

        pred = self.student(imgs)
        pred_gt=self.student(gt_images)
        
        gt_loss_factor=1e-2
        # print(f"target shape {targets.shape}---- pred shape {pred[0].shape} ---- pred_gt shape {pred_gt[0].shape}")
        # Loss
        compute_loss = ComputeLoss(self.student)
        loss, loss_items = compute_loss(pred, targets)  # scaled by batch_size
        loss_gt, _ = compute_loss(pred_gt, targets)  # scaled by batch_size
        # print(f"loss_gt {loss_gt} loss {loss}")
        return loss+loss_gt*gt_loss_factor, loss_items