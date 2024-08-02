import numpy as np
import torch
import torch.nn as nn

def compute_distillation_output_loss(p, t_p, model, dist_loss="l2", T=20, reg_norm=None):
    t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError(
            "reduction must be mean in distillation mode!")

    DboxLoss = nn.MSELoss(reduction="none")
    if dist_loss == "l2":
        DclsLoss = nn.MSELoss(reduction="none")
    elif dist_loss == "kl":
        DclsLoss = nn.KLDivLoss(reduction="none")
    else:
        DclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    DobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        t_pi = t_p[i]
        t_obj_scale = t_pi[..., 4].sigmoid()

        

        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        if not reg_norm:
            print("pi shape: ", pi.shape)
            print("t_pi shape: ", t_pi.shape)
            print("pi[..., 2:4] shape: ", pi[..., 2:4].shape)
            print("t_pi[..., 2:4] shape: ", t_pi[..., 2:4].shape)
            t_lbox += torch.mean(DboxLoss(pi[..., :4],
                                          t_pi[..., :4]) * b_obj_scale)
            
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(
                0).unsqueeze(-2).unsqueeze(-2)
            t_lbox += torch.mean(DboxLoss(pi[..., :2].sigmoid(),
                                          t_pi[..., :2].sigmoid()) * b_obj_scale)
            t_lbox += torch.mean(DboxLoss(pi[..., 2:4].sigmoid(),
                                          t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1,
                                                           1, 1, 1, model.nc)
            if dist_loss == "kl":
                kl_loss = DclsLoss(F.log_softmax(pi[..., 5:]/T, dim=-1),
                                   F.softmax(t_pi[..., 5:]/T, dim=-1)) * (T * T)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:
                t_lcls += torch.mean(DclsLoss(pi[..., 5:],
                                              t_pi[..., 5:]) * c_obj_scale)

        t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['box'] * h['dist']
    t_lobj *= h['obj'] * h['dist']
    t_lcls *= h['cls'] * h['dist']
    bs = p[0].shape[0]  # batch size
    dloss = (t_lobj + t_lbox + t_lcls) * bs
    return dloss

def compute_distillation_feature_loss(s_f, t_f, model, loss,single_layer_only=False):
    distillation_factor=0.01
    h = model.hyp  # hyperparameters
    ft = torch.cuda.FloatTensor if s_f[0].is_cuda else torch.Tensor
    dl_1, dl_2, dl_3 = ft([0]), ft([0]), ft([0])
    bs = s_f[0].shape[0]

    loss_func1 = nn.MSELoss(reduction="mean")
    dl_1 += loss_func1(s_f[0], t_f[0])
    dl_1 *= h['dist'] * distillation_factor


    if(not single_layer_only):
        loss_func2 = nn.MSELoss(reduction="mean")
        loss_func3 = nn.MSELoss(reduction="mean")
        dl_2 += loss_func2(s_f[1], t_f[1])
        dl_3 += loss_func3(s_f[2], t_f[2])
        dl_2 *= h['dist'] * distillation_factor
        dl_3 *= h['dist'] * distillation_factor
        loss += (dl_1 + dl_2 + dl_3) * bs
        return loss
    loss += (dl_1) * bs
    return loss


   
    
    
    