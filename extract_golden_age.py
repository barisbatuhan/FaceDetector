import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Internal Files
from BBTNet.model.retinaface import RetinaFace
from BBTNet.reader.image_reader import read_image
from BBTNet.utils.box_processes import _get_priorboxes, encode_gt_and_get_indices, nms, decode_boxes
from configs import *

parser = argparse.ArgumentParser(description='Face Detector Golden Age Annotation Extractor')
parser.add_argument('--net', default="./weights/final_mixed_r50.pth", help='Net weights for predicting the boxes')
parser.add_argument('--img_path', default='./data/', help='Location to load the image to predict')
parser.add_argument('--log_file', default='./logs/golden_extract_logs.txt', help='Location for logging')
parser.add_argument('--save_path', default='./data/golden_annot/', help='Location for annotation saving')
parser.add_argument('--backbone', default="resnet50", help='Location to save training logs')
parser.add_argument('--partition', default='all', help='It is either 1 or 2 or 3 or all (divides the whole data into 3 and takes one part)')
parser.add_argument('--golden_path', default='/datasets/COMICS/raw_panel_images/', help='Location of the root golden age comic folder')
parser.add_argument('--conf_thold', default=0.55, type=float, help='Confidence threshold for faces')
parser.add_argument('--nms_thold', default=0.2, type=float, help='NMS threshold for faces')

args = parser.parse_args()

model = RetinaFace(args.backbone)
model.load_state_dict(torch.load(args.net))
model = model.to(device)
model.eval()
print("[INFO] Model is loaded!")

comics = os.listdir(args.golden_path)
comics.sort()
clen = len(comics)

if args.partition == "1":
    comics = comics[:clen//3]
elif args.partition == "2":
    comics = comics[clen//3:2*(clen//3)]
elif args.partition == "3": 
    comics = comics[2*(clen//3):]

ctr = 1
for comic in comics:
    print("Processing:", ctr, "/", len(comics), "comic series.")
    if args.log_file is not None:
        with open(args.log_file, "a") as f:
            f.write("Processing:" + str(ctr) + "/" + str(len(comics)) + "comic series.\n")
        
    f = open(args.save_path + comic + ".txt", "w")
    f_none = open(args.save_path + "no_person_frames.txt", "a")
    images = os.listdir(args.golden_path + comic + "/")
    images.sort()
    for image in images:
        path = args.golden_path + comic + "/" + image
        img = read_image(path, None, augment=False, resize_len=[-1, -1]).unsqueeze(0)
        N, C, H, W = img.size()

        with torch.no_grad():
            
            priors = _get_priorboxes(model.num_anchors, model.anchor_info, [W, H])
            class_vals, bbox_vals = model(img)
            bbox_vals = decode_boxes(bbox_vals[0,:,:], priors)
            scores = F.softmax(class_vals, dim=2)[0,:,1]
            
            if scores.max() <= 0.08:
                # no faces are probably available, classifies as background panel
                # for later usage
                f_none.write(path + "\n")
                
            else:
                # confidence filtering
                conf_pass = torch.where(scores > args.conf_thold)[0]
                bbox_vals = bbox_vals[conf_pass,:]
                scores = scores[conf_pass]
                # nms filtering
                nms_pass = nms(scores, bbox_vals, args.nms_thold)
                bbox_vals = bbox_vals[nms_pass,:]
                scores = scores[nms_pass]
                
                for person in range(scores.shape[0]):
                    s = scores[person].cpu().item()
                    x1, y1, x2, y2 = bbox_vals[person,:].cpu().floor().long().numpy()
                    f.write(
                        comic + "/" + image + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(s) + "\n")
                
    ctr += 1
    # frequently closing so that the process can be followed from these files
    f.close()
    f_none.close()