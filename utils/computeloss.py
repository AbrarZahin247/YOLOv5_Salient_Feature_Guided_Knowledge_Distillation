import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

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

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
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
    def __call__(self, p, targets):  # predictions, targets
        with torch.no_grad():
            lcls = torch.zeros(1, device=self.device)  # class loss
            lbox = torch.zeros(1, device=self.device)  # box loss
            lobj = torch.zeros(1, device=self.device)  # object loss
            tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
            all_pboxes = []
            all_indices=[]
            # print(f'p len ==> {len(p)}')
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                # print(f'b size ==> {b.size()}')

                tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                    # Regression
                    pxy = pxy.sigmoid() * 2 - 0.5
                    pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    # pbox=torch.cat(pbox,pbox)
                    # print(f'pbox size ==> {pbox.size()}')
                    # Accumulate predicted boxes per batch
                    # if b not in all_pboxes.keys():
                    all_indices.append(b)
                    all_pboxes.append(pbox)
                    # all_pboxes[b].append(pbox)

            # Concatenate predicted boxes for each batch
            # concatenated_pboxes = torch.cat([torch.stack(all_pboxes[batch]) for batch in sorted(all_pboxes.keys())])
        stacked_tensor_indices = torch.cat([t for t in all_indices], dim=0)
        stacked_tensor_bboxs = torch.cat([t for t in all_pboxes], dim=0)
        # print(stacked_tensor_indices.size())
        # print(stacked_tensor_bboxs.size())
        return stacked_tensor_indices,stacked_tensor_bboxs
    # def __call__(self, p, targets):  # predictions, targets
    #     with torch.no_grad():
    #         lcls = torch.zeros(1, device=self.device)  # class loss
    #         lbox = torch.zeros(1, device=self.device)  # box loss
    #         lobj = torch.zeros(1, device=self.device)  # object loss
    #         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
    #         all_pboxes={}
            
    #         # all_indexes=[]
    #         for i, pi in enumerate(p):  # layer index, layer predictions
    #             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                

    #             # all_indexes.append(b)
    #             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

    #             n = b.shape[0]  # number of targets
    #             if n:
    #                 # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
    #                 pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

    #                 # Regression
    #                 pxy = pxy.sigmoid() * 2 - 0.5
    #                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
    #                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
    #                 if(b not in all_pboxes.keys()):
    #                     all_pboxes[b]=[]
    #                 all_pboxes[b].append(pbox)
    #                 # all_pboxes.append(pbox)
    #     return torch.cat(list(all_pboxes.values()))
        # return tcls, tbox, indices, anchors


    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
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
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            print(f"build target shape ===> {shape}")
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
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