import argparse
import torch
import torch.nn as nn
from models.yolo import Model
from utils.general import intersect_dicts
from copy import deepcopy
import torch.nn.utils.prune as prune



def parse_args():
    parser = argparse.ArgumentParser(description="Load and prune YOLO model")
    parser.add_argument('--saved_name', type=str, default="pruned_model", help="Path to model weights or a URL to pretrained model")
    parser.add_argument('--weights', type=str, required=True, help="Path to model weights or a URL to pretrained model")
    # parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on (cuda or cpu)")
    parser.add_argument('--cfg', type=str, default=None, help="Path to the config file (optional)")
    parser.add_argument('--nc', type=int, default=80, help="Number of classes")
    parser.add_argument('--hyp', type=str, default=None, help="Path to the hyperparameters file (optional)")
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--prune_amount', type=float, default=0.2, help="Amount of pruning (fraction)")
    return parser.parse_args()

def calculate_zero_weights_percentage(model):
    # Get all the weights from the model (parameters)
    total_weights = 0
    zero_weights = 0

    # Iterate through all model parameters (weights)
    for param in model.parameters():
        # Flatten the parameter tensor and count the zero elements
        total_weights += param.numel()
        zero_weights += (param == 0).sum().item()

    # Calculate the percentage of zero weights
    zero_percentage = (zero_weights / total_weights) * 100
    return zero_percentage


def main():
    # Parse arguments
    args = parse_args()

    # Initialize device
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device="cpu"
    # Check if pretrained weights file
    pretrained = args.weights.endswith(".pt")
    print(f"Loading and pruning the model...{pretrained}")
    if pretrained:
        # Handle distributed training (if applicable)
        # if args.local_rank != -1:
        #     with torch_distributed_zero_first(args.local_rank):
        #         args.weights = attempt_download(args.weights)  # Download if not found locally
        # else:
        #     args.weights = attempt_download(args.weights)  # Download if not found locally
        
        # Load the checkpoint
        ckpt = torch.load(args.weights, map_location="cpu")  # Load checkpoint to CPU
        
        # Create model from the config or checkpoint YAML
        model = Model(ckpt["model"].yaml, ch=3, nc=args.nc, anchors=None).to(device)
        
        # Exclude 'anchor' key if anchors are provided
        exclude = ["anchor"] if (args.cfg or args.hyp) and not args.resume else []
        
        # Get the state_dict from the checkpoint
        csd = ckpt["model"].float().state_dict()  # FP32 state_dict
        
        # Intersect the checkpoint dict with the model state_dict
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        
        # Load the state_dict into the model
        model.load_state_dict(csd, strict=False)

        # Apply pruning to the Conv2d layers after loading the weights
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                prune.l1_unstructured(m, name='weight', amount=args.prune_amount)  # Prune the weights
                prune.remove(m, 'weight')  # Make pruning permanent
        
        # Optionally, save the pruned model
        saved_path=args.saved_name + ".pt"
        ckpt = {
                    # "epoch": epoch,
                    # "best_fitness": best_fitness,
                    "model": deepcopy(model).half()
                    # "ema": deepcopy(ema.ema).half(),
                    # "updates": ema.updates,
                    # "optimizer": optimizer.state_dict(),
                    # "opt": vars(opt),
                    # "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    # "date": datetime.now().isoformat(),
                }

        # Calculate and print the percentage of zero weights
        zero_percentage = calculate_zero_weights_percentage(model)
        print(f"Percentage of zero weights: {zero_percentage:.2f}%")

        torch.save(ckpt, saved_path)        
        print(f"{args.prune_amount*100}% of the model pruned successfully.")
    else:
        print("Weights not in .pt format, skipping loading...")

if __name__ == "__main__":
    main()