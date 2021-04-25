import torch
import argparse

from BBTNet.model.retinaface import RetinaFace
from BBTNet.reader.widerface import WF_Data
from BBTNet.reader.icartoonface import ICF_Data
from BBTNet.reader.mixed_data import Mixed_Data
from configs import *

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--start_net', default=None, help='resume net for retraining')
parser.add_argument('--start_epoch', default=1, type=int, help='resume iter for retraining')
parser.add_argument('--num_epochs', default=60, type=int, help='until which epoch to train')
parser.add_argument('--save_dir', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--log_file', default=None, help='Location to save training logs')
parser.add_argument('--backbone', default="resnet50", help='Location to save training logs')
parser.add_argument('--dataset', default="mixed", help='resume net for retraining')

args = parser.parse_args()

backbone = args.backbone
if args.dataset == "wf":
    tr_path = wf_path + "train/"
    label_path = wf_labels_path + "train/label.txt"
    d_det = WF_Data(tr_path, label_path, batch_size=rf_bs, augment=True, img_sizes=[640, 640])
elif args.dataset == "icf":
    tr_path = ICF_tr_path
    label_path = ICF_tr_labels_path + "icartoonface_dettrain.csv"
    d_det = ICF_Data(tr_path, label_path, batch_size=rf_bs, augment=True, img_sizes=[640, 640])
elif args.dataset == "mixed":
    wf_tr_path = wf_path + "train/"
    wf_label_path = wf_labels_path + "train/label.txt"
    icf_tr_path = ICF_tr_path
    icf_label_path = ICF_tr_labels_path + "icartoonface_dettrain.csv"
    d_det = Mixed_Data(
        wf_tr_path, wf_label_path, icf_tr_path, icf_label_path, 
        batch_size=rf_bs, augment=True, img_sizes=[640, 640]
    )

print("[INFO] Data is loaded!")

model = RetinaFace(backbone=backbone)
if args.start_net is not None:
    model.load_state_dict(torch.load(args.start_net))
model = model.to(device)
print("[INFO] Model is loaded!")

model.train_model(
    d_det, args.num_epochs, weight_decay=weight_decay, start_epoch=args.start_epoch,
    init_lr=args.lr, log_file=args.log_file, save_dir=args.save_dir
)