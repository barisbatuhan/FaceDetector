# External Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Internal Files
from BBTNet.components.backbone import Backbone
from BBTNet.components.fpn import FeaturePyramid
from BBTNet.components.heads import HeadGetter
from BBTNet.components.context_module import ContextModule
from BBTNet.utils.box_processes import _get_priorboxes, encode_gt_and_get_indices, nms, decode_boxes
from configs import *

class RetinaFace(torch.nn.Module):
    def __init__(
        self, 
        backbone="resnet50",
        img_size=[640, 640]
    ):
        super(RetinaFace, self).__init__()
        self.num_anchors = 3
        self.anchor_info = [ 
            {"stride": 4, "anchors": [16, 20.16, 25.40]},
            {"stride": 8, "anchors": [32, 40.32, 50.80]},
            {"stride": 16, "anchors": [64, 80.63, 101.59]},
            {"stride": 32, "anchors": [128, 161.26, 203.19]},
            {"stride": 64, "anchors": [256, 322.54, 406.37]}
        ]

        if img_size is None or img_size[0] < 0 or img_size[1] < 0:
            self.priors = None # no need to store priors if only prediction will be made
        else:
            self.priors = _get_priorboxes(self.num_anchors, self.anchor_info, img_size)
        
        self.backbone = Backbone(backbone=backbone) 
        in_size = self.backbone.in_sizes
        self.fpn = FeaturePyramid(in_sizes=in_size, out_size=256)
        self.context_module = ContextModule(in_size=256, out_size=256)
        self.heads2 = nn.ModuleList([
            HeadGetter(256, 2),
            HeadGetter(256, 4),
        ])
      
    def forward(self, x):
        p_vals = self.backbone(x)
        p_vals = self.fpn(p_vals)
        p_vals = self.context_module(p_vals)
        class_vals = self.heads2[0](p_vals)
        bbox_vals = self.heads2[1](p_vals)
        return class_vals, bbox_vals
    
        
    def get_loss(self, x, y):
        class_vals, bbox_vals = self(x); cls_res = F.softmax(class_vals, dim=2)
        N, P, T = class_vals.shape
        lposcls = 0; lnegcls = 0; lbox = 0; clsNeg = 0; boxN = 0
        
        for n in range(N):          
            gt, pos_idx, neg_idx = encode_gt_and_get_indices(
                y[n], self.priors, cls_res[n,:,0], pos_iou, neg_iou)
            
            if pos_idx is None or len(pos_idx) < 1:
                continue

            # cls loss
            pos_lcls = F.cross_entropy(
                class_vals[n,pos_idx,:], 
                torch.ones(len(pos_idx), dtype=torch.long).to(device), 
                reduction='sum'
            )
            lposcls += pos_lcls

            neg_lcls = F.cross_entropy(
                class_vals[n,neg_idx,:], 
                torch.zeros(len(neg_idx), dtype=torch.long).to(device), 
                reduction='sum'
            )
            lnegcls += neg_lcls
            clsNeg += len(neg_idx)
            
            # box loss
            lbox += F.smooth_l1_loss(bbox_vals[n,pos_idx,:], gt[:,0:4], reduction='sum')
            boxN += len(pos_idx)

        boxN = max(1, boxN); clsNeg = max(1, clsNeg)
        return (lposcls + lnegcls) / boxN, lbox/boxN

    def train_model(
        self, det_data, epochs, start_epoch=1, weight_decay=0.0005, 
        init_lr=0.001, lr_changes=[3, 25, 40], log_file=None, save_dir=None
    ):
        optimizer = optim.SGD(self.parameters(), lr=init_lr, momentum=0.9, weight_decay=weight_decay)
        for e in range(start_epoch, epochs+1):
            lr = self.adjust_learning_rate(optimizer, e, init_lr, lr_changes)
            x, y = det_data.forward(restart=True)
            curr_batch = 0; total_imgs = len(det_data.imgs_path)
        
            while x is not None and y is not None:
                curr_batch += x.shape[0]
                optimizer.zero_grad()
                lcls, lbox = self.get_loss(x, y)
                total_loss = lcls + lambda1 * lbox 
                total_loss.backward()
                optimizer.step()
            
                self.print_losses(
                    e, curr_batch, total_imgs, lr, lcls.data.item(), lbox.data.item(), log_file=log_file
                )
                x, y = det_data.forward()
                         
            if save_dir is not None:
                self.save_model(save_dir + "model_epoch" + str(e) + ".pth")
    
    
    def predict_image(self, x, nms_thold=0.4, conf_thold=0.5, topk=5000, keep_topk=None, filter=True):
        N, C, H, W = x.shape
        priors = _get_priorboxes(self.num_anchors, self.anchor_info, [W, H])
        with torch.no_grad():
            class_vals, bbox_vals = self(x)
        bbox_vals = decode_boxes(bbox_vals[0,:,:], priors)
        scores = F.softmax(class_vals, dim=2)[0,:,1]
        
        if not filter:
            return scores.cpu().data.numpy(), bbox_vals.cpu().data.numpy()
        
        # confidence filtering
        conf_pass = torch.where(scores > conf_thold)[0]
        bbox_vals = bbox_vals[conf_pass,:]
        scores = scores[conf_pass]
        # Keep top-k
        tops = torch.argsort(scores, descending=True)[:topk]
        bbox_vals = bbox_vals[tops,:]
        scores = scores[tops]
        # nms filtering
        nms_pass = nms(scores, bbox_vals, nms_thold)
        bbox_vals = bbox_vals[nms_pass,:]
        scores = scores[nms_pass]
        # last keep-topk filtering
        if keep_topk is not None and len(scores) > keep_topk:
            most_tops = torch.argsort(scores, descending=True)[:keep_topk]
            bbox_vals = bbox_vals[most_tops,:]
            scores = scores[most_tops]
        
        return scores.cpu().data.numpy(), bbox_vals.cpu().data.numpy()

    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

        
    def adjust_learning_rate(self, optimizer, epoch, init_lr, lr_changes):
        if epoch < lr_changes[0]:
            lr = init_lr
        elif lr_changes[0] <= epoch < lr_changes[1]:
            lr = 10 * init_lr
        elif lr_changes[1] <= epoch < lr_changes[2]:
            lr = init_lr
        else:
            lr = init_lr / 10
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
       
        return lr
    
    
    def print_losses(self, epoch, batch, total, lr, lcls, lbox, log_file=None):
        to_print = "E: " + str(epoch) 
        to_print +=  " & B: " + str(batch) + "/" + str(total)
        to_print += " & LR: " + str(lr)[:min(6, len(str(lr)))]
        to_print += " ---> Cls: " + str(lcls)[:min(6, len(str(lcls)))]
        to_print += " | Box: " + str(lbox)[:min(6, len(str(lbox)))] 
        print(to_print)
        if log_file is not None:
            f = open(log_file, "a")
            f.write(to_print + "\n")
            f.close()