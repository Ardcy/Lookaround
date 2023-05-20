import torch.nn.init as init
import math
import time
import sys
import os
import logging
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from lookaround import *


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def set_random_seed(seed_num):
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "Log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def make_optimizer(cfg, net):
    if cfg.SOLVER.OPTIMIZER_NAME == "Lookaround":
        return Lookaround(net.parameters(), lr=cfg.SOLVER.LR, momentum=0.9, weight_decay=5e-4, head_num=cfg.BUILD_TRANSFORM_NUM, frequence=cfg.SOLVER.FREQUENCY)
    if cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        return optim.SGD(net.parameters(), lr=cfg.SOLVER.LR, momentum=0.9, weight_decay=5e-4)
    if cfg.SOLVER.OPTIMIZER_NAME == "Adam":
        return torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.Adam_LR, betas=(cfg.SOLVER.Adam_Beta1, cfg.SOLVER.Adam_Beta2), weight_decay=cfg.SOLVER.Adam_weight_decay)


def make_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'MultiStepLR':
        MILESTONES = cfg.SOLVER.SCHEDULER_MVALUE
        if cfg.SOLVER.SCHEDULER_MVALUE[0] != 0:
            MILESTONES = cfg.SOLVER.SCHEDULER_MVALUE
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=cfg.SOLVER.SCHEDULER_GAMMA)
    if cfg.SOLVER.SCHEDULER == 'Cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.MAX_EPOCHS, eta_min=cfg.SOLVER.MIN_LR)


def make_loss(cfg):
    if cfg.SOLVER.LOSS == 'CrossEntropy':
        return nn.CrossEntropyLoss()
