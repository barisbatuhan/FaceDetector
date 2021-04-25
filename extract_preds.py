import torch
import argparse
import os

from BBTNet.model.retinaface import RetinaFace
from BBTNet.reader.image_reader import read_image, get_PIL_image
from BBTNet.utils.visualize import draw_boxes
from BBTNet.reader.widerface import WF_Data
from configs import *

parser = argparse.ArgumentParser(description='Retinaface Image Prediction')
parser.add_argument('--net', default=None, help='Net weights for predicting the boxes')
parser.add_argument('--backbone', default="resnet50", help='Location to save training logs')
parser.add_argument('--data', default=0, type=int, help='0 for widerface, 1 for icartoonface, 2 for manga109 extraction')
parser.add_argument('--save_dir', default='./data/', help='Location (for widerface) / File (for others) to save the predictions')

args = parser.parse_args()

if args.net is None:
    print("[ERROR] A pretrained network weight has to be given with the flag --net")
    raise

model = RetinaFace(args.backbone)
model.load_state_dict(torch.load(args.net))
model = model.to(device)
model.eval()
print("[INFO] Model is loaded!")

if args.data == 0:
    root_dir = ""
    d = WF_Data(wf_path + "val/", wf_labels_path + "val/label.txt", batch_size=1, augment=False, img_sizes=[-1,-1])
    img_paths = d.imgs_path
    nms_thold = 0.4
    conf_thold = 0.02
    topk = 5000
    keep_topk = None
elif args.data == 1:
    root_dir = ICF_val_path
    img_paths = os.listdir(ICF_val_path)
    nms_thold = 0.55
    conf_thold = 0.08
    topk = 1000
    keep_topk = 100
elif args.data == 2:
    root_dir = manga109_path + "val_imgs/"
    img_paths = os.listdir(manga109_path + "val_imgs/")
    nms_thold = 0.55
    conf_thold = 0.08
    topk = 1000
    keep_topk = 100
elif args.data == 3:
    root_dir = manga109_path + "test_imgs/"
    img_paths = os.listdir(manga109_path + "test_imgs/")
    nms_thold = 0.55
    conf_thold = 0.08
    topk = 1000
    keep_topk = 100

if args.data > 0:
    f = open(args.save_dir, "a")

ctr = 1
for img_path in img_paths:
    if ".ipynb_checkpoints" in img_path:
        continue
    # image prediction
    img = read_image(root_dir + img_path, None, augment=False, resize_len=[-1, -1]).unsqueeze(0).to(device)
    cls, box = model.predict_image(img, nms_thold=nms_thold, conf_thold=conf_thold, topk=topk, keep_topk=keep_topk)
    num_preds = cls.shape[0]
    print("Finished:", ctr, "/", len(img_paths), "--> Found:", num_preds)
    ctr += 1
    
    if args.data > 0:
        for i in range(num_preds):
            x1, y1, x2, y2 = box[i,:].astype(int)
            f.write(img_path + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ",face," + str(cls[i]) + "\n")
    else:
        first_slash = img_path.rfind("/")
        filename = img_path[first_slash+1:-4]
        sec_slash = img_path[:first_slash].rfind("/")
        event = img_path[sec_slash+1:first_slash]
        # Saving to a file
        folder_path = args.save_dir + event + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(folder_path + filename + ".txt", "w")
        f.write(filename + "\n")
        f.write(str(num_preds) + "\n")
        for person in range(num_preds):
            x1, y1, x2, y2 = box[person,:].astype(int)
            w = x2 - x1; h = y2 - y1
            f.write(str(x1) + " " + str(y1) + " " + str(w) + " " + str(h) + " " + str(cls[person]) + "\n")
        f.close()

if args.data > 0:
    f.close()
