from __future__ import print_function
import subprocess
import os
import pdb
import random
from datetime import datetime
curDT = datetime.now()
date_time = curDT.strftime("%Y-%m-%d %H:%M")
from utils.logger import GOATLogger
from copy import deepcopy

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
# import torch.nn.functional as F
import torch.distributed as dist

from .network.get_network import get_network
from utils.common_util import AverageMeter, intersectionAndUnionGPU

seed=0
pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False 

class net_pointmixer(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        
        # ------------
        # save_hyperparameters
        # ------------
        self.save_hyperparameters("args")
        args = self.hparams['args']        
        self.MYCHECKPOINT = deepcopy(args.MYCHECKPOINT)
        self.voxel_size = deepcopy(args.voxel_size)
        self.ignore_label = int(args.ignore_label) # 255
        self.classes = int(args.classes)
        self.print_freq = int(args.print_freq)
        self.optim = deepcopy(args.optim)

        self.train_batch = int(args.train_batch)
        self.val_batch = int(args.val_batch)

        self.distributed_backend = str(args.distributed_backend)

        # ------------
        # model
        # ------------        
        self.model = get_network(args)

        # ------------
        # metrics
        # ------------
        self.resetMetrics()

        # ------------
        # logger
        # ------------
        if not bool(args.off_text_logger): # FIXME. pytorch-lightning does not support text logger.
            mode = 'train' if bool(args.on_train) else 'test'
            self.text_logger = GOATLogger(
                mode=mode, 
                save_root=args.MYCHECKPOINT,
                log_freq=0,
                base_name=f'log_{mode}_{date_time}',
                n_iterations=0,
                n_eval_iterations=0)
        else:
            self.text_logger = None
            
                
    def resetMetrics(self):
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()
        self.nvox_meter = AverageMeter()

    def forward(self, data_dict):
        pred_dict = {}
        loss_dict = {}

        with torch.no_grad():
            coord = data_dict['coord']
            feat = data_dict['feat']
            target = data_dict['target']
            offset = data_dict['offset']
        
        output = self.model([coord, feat, offset])
        pred_dict['output'] = output
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)(output, target)
        loss_dict['loss'] = loss

        return pred_dict, loss_dict
    
    @torch.no_grad()
    def calcMetrics(self, batch_idx, outputs, data_dict, logName='train/accuracy'):
        output = outputs['output'].detach()
        coord = data_dict['coord']
        target = data_dict['target']

        output = output.max(1)[1]
        n = coord.size(0)        
        count = target.new_tensor([n], dtype=torch.long)
        n = count.item()
        intersection, union, target = \
            intersectionAndUnionGPU(output, target, self.classes, self.ignore_label)
        if self.distributed_backend != 'dp':
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection = intersection.cpu().detach().numpy()
        union = union.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)

        accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)
        if batch_idx % self.print_freq == 0:
            self.log(logName, accuracy)

    def training_step(self, data_dict, batch_idx):
        outputs, losses = self.forward(data_dict)

        if (self.global_rank == 0) and (self.global_step % 1000 == 0):

            nvox = float(data_dict['coord'].size(0)) / float(self.train_batch)
            
            if self.logger is not None:
                self.logger.experiment["nvox_train"].log(nvox)
                self.logger.experiment["current_epoch_train"].log(self.current_epoch)
                self.logger.experiment["global_step_train"].log(self.global_step)
                self.logger.experiment["lr_train"].log(self.scheduler.get_last_lr())
            if self.text_logger is not None:
                str_to_print = (
                    f'train : epoch[{self.current_epoch:d}], steps[{self.global_step:d}] lr[{self.scheduler.get_last_lr()[0]:.4f}] | '
                    f'nvox[{int(nvox):d}], '
                )
                self.text_logger.loginfo(str_to_print)
            print("TRAIN: epoch[%d], global_step[%d]: \n"%(self.current_epoch, self.global_step))
    
        self.resetMetrics() ######### WRONG

        loss = sum(losses.values())
        return loss

    @torch.no_grad()
    def training_epoch_end(self, outputs):
        # train_step -> val_step -> val_epoch_end -> train_epoch_end 
        self.scheduler.step()

    @torch.no_grad()
    def validation_step(self, data_dict, batch_idx):
        with torch.no_grad():
            outputs, losses = self.forward(data_dict)
            self.calcMetrics(batch_idx, outputs, data_dict, logName='val/accuracy')
         
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
            accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
            
            self.log("mIoU_val", mIoU)
            if self.global_rank == 0:
                if self.logger is not None:
                    self.logger.experiment["mIoU_val"].log(mIoU)
                    self.logger.experiment["mAcc_val"].log(mAcc)
                    self.logger.experiment["allAcc_val"].log(allAcc)
                    self.logger.experiment["epoch_log"].log(self.current_epoch)
                    self.logger.experiment["lr_log"].log(self.scheduler.get_last_lr())

                    self.logger.experiment["current_epoch_val"].log(self.current_epoch)
                    self.logger.experiment["global_step_val"].log(self.global_step)
                    self.logger.experiment["lr_val"].log(self.scheduler.get_last_lr())
                if self.text_logger is not None:
                    str_to_print = (
                        f'val : epoch[{self.current_epoch:d}], steps[{self.global_step:d}] lr[{self.scheduler.get_last_lr()[0]:.4f}] | '
                        f'mIoU_val[{mIoU:.2f}], '
                        f'mAcc_val[{mAcc:.2f}], '
                        f'allAcc_val[{allAcc:.2f}], '
                    )
                    self.text_logger.loginfo(str_to_print)

            print("VAL: epoch[%d]: mIoU[%.3f] \n"%(self.current_epoch, mIoU))
            self.resetMetrics()

    @torch.no_grad()
    def test_step(self, data_dict, batch_idx):
        with torch.no_grad():
            outputs, losses = self.forward(data_dict)
            self.calcMetrics(batch_idx, outputs, data_dict, logName='test/accuracy')

            pred_idx = data_dict['pred_idx']
            pred_part = outputs['output']
            self.pred[pred_idx, :] += pred_part
            
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        with torch.no_grad():
            iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
            accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
            
            if self.global_rank == 0:
                if self.logger is not None:
                    self.logger.experiment["mIoU_test_per_scene"].log(mIoU)
                    self.logger.experiment["mAcc_test_per_scene"].log(mAcc)
                    self.logger.experiment["allAcc_test_per_scene"].log(allAcc)

                    str_to_log = "TEST_per_scene: epoch[%d]: mIoU[%.3f], mAcc[%.3f], allAcc[%.3f] \n"%(self.current_epoch, mIoU, mAcc, allAcc)
                    print(str_to_log)
                    self.logger.experiment['logs'].log(str_to_log)
                if self.text_logger is not None:
                    str_to_print = (
                        f'test | '
                        f'mIoU_test_per_scene[{mIoU:.4f}], '
                        f'mAcc_test_per_scene[{mAcc:.4f}], '
                        f'allAcc_test_per_scene[{allAcc:.4f}], '
                    )
                    self.text_logger.loginfo(str_to_print)

        self.resetMetrics()

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        args = self.hparams['args']

        if self.optim in ['Adam', 'AdamW', 'NAdam'] :
            kwargs = \
                {
                    'params': self.parameters(),
                    'lr': float(args.lr), 
                    'weight_decay': args.weight_decay,
                }
            optimizer = getattr(torch.optim, self.optim)(**kwargs)
            milestones = [int(args.epochs*ratio) for ratio in args.schedule]
            for _ in range(5):
                print(">> lr schedule gamma[%f]"%(args.lr_GAMMA), milestones)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=args.lr_GAMMA)

        elif self.optim in ['SGD', 'ASGD']:
            optimizer = getattr(torch.optim, self.optim)(
                self.parameters(), lr=float(args.lr), momentum=args.momentum, weight_decay=args.weight_decay)
            milestones = [int(args.epochs*ratio) for ratio in args.schedule]
            for _ in range(5):
                print(">> lr schedule gamma[%f]"%(args.lr_GAMMA), milestones)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=args.lr_GAMMA)
                
        else:
            raise NotImplemented
        optimizers.append(optimizer)

        return optimizers, schedulers
