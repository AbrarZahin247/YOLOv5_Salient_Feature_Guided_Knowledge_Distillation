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
    def __init__(self, device, anchors, nl, na, nc, hyp):
        self.device = device
        self.anchors = anchors
        self.nl = nl
        self.na = na
        self.nc = nc
        self.hyp = hyp
    
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
        
        return lbox,lobj,lcls

    def __call__(self, p, tp, targets):  # predictions, teacher predictions, targets
        with torch.no_grad():
            _, tbox, indices, anchors = self.build_targets(p, targets)  # targets
            _, tpbox, tp_indices, tp_anchors = self.build_targets(tp, targets)  # teacher targets

            student_boxes = []
            teacher_boxes = []
            target_boxes = []

            for i in range(self.nl):  # for each prediction layer
                b, a, gj, gi = indices[i]  # student indices
                tb, ta, tgj, tgi = tp_indices[i]  # teacher indices

                if b.shape[0] > 0:
                    pxy, pwh, _, _ = p[i][b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                    pxy = pxy.sigmoid() * 2 - 0.5
                    pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                    student_boxes.append(torch.cat((pxy, pwh), 1))  # student predicted box

                if tb.shape[0] > 0:
                    tpxy, tpwh, _, _ = tp[i][tb, ta, tgj, tgi].split((2, 2, 1, self.nc), 1)
                    tpxy = tpxy.sigmoid() * 2 - 0.5
                    tpwh = (tpwh.sigmoid() * 2) ** 2 * tp_anchors[i]
                    teacher_boxes.append(torch.cat((tpxy, tpwh), 1))  # teacher predicted box

                if b.shape[0] > 0:
                    target_boxes.append(tbox[i])

            if len(student_boxes) > 0 and len(teacher_boxes) > 0 and len(target_boxes) > 0:
                student_boxes = torch.cat(student_boxes, dim=0)
                teacher_boxes = torch.cat(teacher_boxes, dim=0)
                target_boxes = torch.cat(target_boxes, dim=0)

                student_iou, teacher_iou = self.compute_overlap(target_boxes, student_boxes, teacher_boxes)
                print(student_iou,teacher_iou)
                
            lcls = torch.zeros(1, device=self.device)  # class loss
            lbox = torch.zeros(1, device=self.device)  # box loss
            lobj = torch.zeros(1, device=self.device)  # object loss
            # return student_iou, teacher_iou
            return (lbox, lobj, lcls),torch.cat((lbox, lobj, lcls)).detach()
            # else:
            #     return None, None

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
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
                t = targets[0:1]  # Ensure non-empty tensor
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

    def box_iou(self, box1, box2):
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        
        area1 = box_area(box1)
        area2 = box_area(box2)

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        return inter / (area1[:, None] + area2 - inter)

    def compute_overlap(self, targets, student_predictions, teacher_predictions):
        student_iou = self.box_iou(targets, student_predictions)
        teacher_iou = self.box_iou(targets, teacher_predictions)
        return student_iou, teacher_iou