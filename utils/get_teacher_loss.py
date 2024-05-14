import torch
from models.teacheryolo import Model
from utils.computeloss import ComputeLoss
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)

def get_model_performance(dataset,names,hyp,nc,device,weights,imgs,targets):
    ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    # exclude = ["anchor"] if (hyp.get("anchors")) else []  # exclude keys
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    
    
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    
    compute_teacher_loss=ComputeLoss(model)
    pred=model(imgs)
    # print(f"teacher model prediction size ===> {len(pred)}")
    # print(f"teacher model prediction size ===> {pred[0]}")
    tech_cls, tech_box, tech_indices, tech_anchors=compute_teacher_loss(pred,targets.to(device))
    # print(f"teacher clas ===> {tech_cls}")
    return tech_cls, tech_box, tech_indices, tech_anchors