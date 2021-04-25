import torch
import numpy as np
from math import ceil

from configs import *

def encode_gt_and_get_indices(gt, priors, losses, pos_thold, neg_thold):
    
    # getting positive anchor box indices
    iou_vals = iou(gt[:,0:4], _to_min_max_form(priors))
    pos_pairs = torch.where(iou_vals >= pos_thold)
    prior_idx = pos_pairs[1]; gt_idx = pos_pairs[0]

    num_poses = len(prior_idx)
    if num_poses == 0:
        return None, None, None
    
    pos_gt = torch.zeros(num_poses, 15)
    pos_gt = gt[gt_idx, :]
    
    # getting negative box indices with the most loss values
    max_prior_vals, max_prior_idx = iou_vals.max(0, keepdim=True)
    neg_indices = torch.where(max_prior_vals <= neg_thold)[1]
    neg_cnt = ohem_ratio * num_poses
    neg_indices = neg_indices[torch.sort(losses[neg_indices]).indices[0:neg_cnt]]
    
    # filtering w.r.t. the selected positive indices
    pos_gt[:,0:4] = _to_center_length_form(pos_gt[:,0:4])
    selected_priors = priors[prior_idx,:]
    
    # box conversion to the target format
    pos_gt[:,0:2] = (pos_gt[:,0:2] - selected_priors[:,0:2]) / (variances[1] * selected_priors[:,2:4])
    pos_gt[:,2:4] = torch.log(pos_gt[:,2:4]/selected_priors[:,2:4]) / variances[0]

    return pos_gt, prior_idx, neg_indices

def nms(scores, points, thold):
    order = torch.argsort(scores, descending=True)
    keep = []
    for i in order:
        if keep == []:
            keep.append(i.cpu().data.item())
        else:
            iou_vals = iou(points[i:i+1,:], points[keep,:])
            vals = iou_vals.max(1, keepdim=True).values[0]
            if vals[0] <= thold:
                keep.append(i.cpu().data.item())
    return np.array(keep)

def _get_priorboxes(num_anchors, anchor_info, img_size):
    feature_maps = [(ceil(img_size[0]/scale["stride"]), ceil(img_size[1]/scale["stride"]))  for scale in anchor_info]
    num_proposals = num_anchors * sum([i[0]*i[1] for i in feature_maps])
    anchors = np.zeros((num_proposals, 4))
    
    counter = 0
    for idx, f in enumerate(feature_maps):
        scaler = anchor_info[idx]["stride"]
        for h in range(f[1]):
            cy = (h + 0.5) * scaler
            for w in range(f[0]):
                cx = (w + 0.5) * scaler
                for s in anchor_info[idx]["anchors"]:
                    anchors[counter,:] = [cx, cy, s, s]
                    counter += 1
    
    priors = torch.Tensor(anchors).view(-1, 4)
    if run_gpu:
        priors = priors.to(device)
    return priors

def iou(box_a, box_b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def decode_boxes(box, priors):
    centers = box[:,0:2] * variances[1] * priors[:,2:4] + priors[:,0:2]
    lengths = priors[:,2:4] * torch.exp(box[:,2:4] * variances[0])
    boxes = torch.cat((centers, lengths), dim=1)
    return _to_min_max_form(boxes)


def _to_min_max_form(boxes):
    converted = (boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2)
    return torch.cat(converted, dim=1)


def _to_center_length_form(boxes):
    converted = ((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2])
    return torch.cat(converted, dim=1)  
