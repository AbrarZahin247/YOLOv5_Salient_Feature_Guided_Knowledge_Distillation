# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def imitation_loss(teacher, student, mask):
    if student is None or teacher is None:
        return 0
    # print(teacher.shape, student.shape, mask.shape)
    diff = torch.pow(student - teacher, 2) * mask
    diff = diff.sum() / mask.sum() / 2

    return diff



def feature_mse(feature_map1, feature_map2):
        loss=0
        if(feature_map1 is not None and feature_map2 is not None):
            # loss = F.binary_cross_entropy_with_logits(F.sigmoid(feature_map1), F.sigmoid(feature_map2))
            loss = F.mse_loss(feature_map1, feature_map2)
        return loss

def feature_cross_entropy(feature_map1, feature_map2):
        loss=0
        if(feature_map1 is not None and feature_map2 is not None):
            # loss = F.binary_cross_entropy_with_logits(F.sigmoid(feature_map1), F.sigmoid(feature_map2))
            loss = F.binary_cross_entropy_with_logits(feature_map1, feature_map2)
        return loss

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        
    def xywh_to_xyxy(self,boxes, image_size=640):
        """
        Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
        
        Args:
            boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes in (x, y, w, h) format.
            image_size (tuple): Tuple containing the width and height of the image.
            
        Returns:
            torch.Tensor: Tensor of shape (N, 4) containing bounding boxes in (x1, y1, x2, y2) format.
        """
        width, height = image_size,image_size
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Clamp values to be within image boundaries
        x1 = torch.clamp(x1, 0, width)
        y1 = torch.clamp(y1, 0, height)
        x2 = torch.clamp(x2, 0, width)
        y2 = torch.clamp(y2, 0, height)
    
        return torch.stack((x1, y1, x2, y2), dim=1)
        
    # def get_gt_bbox(self,p,targets):
    #     _, tboxs, __, ___ = self.build_targets(p, targets) ## tcls, tboxs, indices, anchors
    #     return self.xywh_to_xyxy(boxes=tboxs)
    
    def get_lcls_lbox_lobj(self,p,tcls, tbox, indices, anchors):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]
        
        return lbox,lobj,lcls,bs
            
    # def __call__(self,p,targets,student=None, teacher=None, teacher_accepted_batch=None):  # predictions, targets, model
        
    #     tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        
    #     ## teachers predictions
        
    #     # print(f"teacher batch ===> {len(teacher_accepted_batch[0])}")
    #     lbox,lobj,lcls=self.get_lcls_lbox_lobj(p,tcls, tbox, indices, anchors)
    #     roi_loss=torch.zeros(1, device=self.device)
    #     distillation_factor=0.5
    #     if(teacher_accepted_batch!=None):
    #         for index,single_item in enumerate(teacher_accepted_batch):
    #             if(single_item==1):
    #                 roi_loss+=torch.tensor(feature_mse(student[index],teacher[index]))
    #         roi_loss=roi_loss/sum(teacher_accepted_batch)
    #         # roi_inv_loss=feature_mse(student_inv,teacher_inv)
    #         # total_loss=foward_factor*roi_loss+backward_factor*roi_inv_loss
    #     total_loss=distillation_factor*roi_loss+lbox+lobj+lcls
    #     return total_loss, torch.cat((lbox, lobj, lcls)).detach()
    
    def __call__(self, p, targets, student=None, teacher=None,teacher_loss=None,training=False):  # predictions, targets, model
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Get teacher's predictions
        lbox, lobj, lcls,std_bs = self.get_lcls_lbox_lobj(p, tcls, tbox, indices, anchors)
        base_losses=lbox + lobj + lcls
        batch_multiplied_std_base_loss=base_losses*std_bs
        roi_loss = torch.tensor(0.0, device=self.device)
        distillation_factor = 0.1
        tolerance_factor=0.1
        # print(f"==> teacher and student loss >> {teacher_loss}--{batch_multiplied_std_base_loss}")
        roi_loss = torch.tensor(feature_mse(student, teacher), device=self.device)
        roi_loss = roi_loss.float().mean()
        
        # if teacher_accepted_batch is not None:
        #     teacher_accepted_batch_tensor = torch.tensor(teacher_accepted_batch, device=self.device)
        #     selected_teacher_indices = torch.nonzero(teacher_accepted_batch_tensor).squeeze()

        #     if selected_teacher_indices.numel() > 0:  # Check if selected indices tensor has elements
        #         student_selected = torch.index_select(student, 0, selected_teacher_indices)
        #         teacher_selected = torch.index_select(teacher, 0, selected_teacher_indices)
        
        if(training and batch_multiplied_std_base_loss-teacher_loss>tolerance_factor):
            total_loss = (distillation_factor * roi_loss + base_losses)*std_bs
        else:
            total_loss=batch_multiplied_std_base_loss
        return total_loss, torch.cat((lbox, lobj, lcls)).detach()


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
