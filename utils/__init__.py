# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Logging utils
"""

import os
from pathlib import Path

import torch

from utils.general import colorstr
from utils.plots import plot_results

# Define the loggers that are available
LOGGERS = {'csv': None, 'tb': None}  # Only local loggers

try:
    from torch.utils.tensorboard import SummaryWriter
    LOGGERS['tb'] = SummaryWriter  # TensorBoard
except ImportError:
    pass


class Loggers:
    """
    Handles local logging for CSV and TensorBoard.
    All Weights & Biases (wandb) and Comet.ml code has been removed.
    """
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=('csv', 'tb')):
        self.save_dir = Path(save_dir)
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        # The keys (columns) for the results.csv file
        self.keys = [
            'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/gate_loss',  # train loss
            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/gate_loss',  # val loss
            'x/lr0', 'x/lr1', 'x/lr2']  # params
        
        # Initialize loggers
        for k in self.include:
            if k == 'tb' and LOGGERS['tb']:
                self.tb = LOGGERS['tb'](str(self.save_dir))
            elif k == 'csv':
                self.csv_file = self.save_dir / 'results.csv'
                if not self.csv_file.exists():
                    with open(self.csv_file, 'w') as f:
                        f.write(','.join(['epoch'] + self.keys) + '\n')

        # The train.py script expects this attribute to exist, even if it's not used.
        self.remote_dataset = False


    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        """
        Callback runs on fit epoch end.
        Args:
            vals (list): List of key metrics and losses.
            epoch (int): Current epoch number.
            best_fitness (float): The best fitness value seen so far.
            fi (float): The fitness value for the current epoch.
        """
        # Log to TensorBoard
        if 'tb' in self.include and hasattr(self, 'tb') and self.tb:
            x = {k: v for k, v in zip(self.keys, vals)}
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
                
        # Log to CSV
        if 'csv' in self.include:
            # Add epoch to the beginning of vals for logging
            log_row = [epoch] + vals
            with open(self.csv_file, 'a') as f:
                f.write(('%20.5g,' * len(log_row) % tuple(log_row)).rstrip(',') + '\n')


    def on_train_end(self, last, best, epoch, results):
        """
        Callback runs on training end.
        """
        # Plot results.csv if it exists
        if 'csv' in self.include and self.csv_file.exists():
            plot_results(file=self.csv_file)

    # All other callbacks are not needed for local logging, so we define empty placeholders
    def on_pretrain_routine_end(self, labels, names):
        pass

    def on_train_start(self, start_epoch, best_fitness):
        pass

    def on_train_batch_end(self, model, ni, imgs, targets, paths, list_mloss):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_val_batch_end(self, pred, si, path, names, im, out):
        pass

    def on_val_end(self, names, plots):
        pass
        
    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        pass
