import torch
from models.teacheryolo import Model
from utils.singlecomputeloss import ComputeLoss
from utils.general import (labels_to_class_weights)

def single_teacher_loss(dataset,names,hyp,nc,device,teacher_weight,given_images,targets):
    # ckpt = torch.load(given_weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    # model = Model(ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    # # exclude = ["anchor"] if (hyp.get("anchors")) else []  # exclude keys
    # csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    # # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    # model.load_state_dict(csd, strict=False)  # load
    
    teacher_ckpt = torch.load(teacher_weight, map_location=device) 
    teacher_model = Model(teacher_ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    
    
    teacher_model.nc = nc  # attach number of classes to model
    teacher_model.hyp = hyp  # attach hyperparameters to model
    teacher_model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    teacher_model.names = names
    
    
    compute_teacher_loss=ComputeLoss(teacher_model)
    pred=teacher_model(given_images)
    # print(f"teacher model prediction size ===> {len(pred)}")
    # print(f"teacher model prediction size ===> {pred[0]}")
    return compute_teacher_loss(pred,targets.to(device))