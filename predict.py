import torch
import argparse

from BBTNet.model.retinaface import RetinaFace
from BBTNet.reader.image_reader import read_image, get_PIL_image
from BBTNet.utils.visualize import draw_boxes
from configs import *

parser = argparse.ArgumentParser(description='Retinaface Image Prediction')
parser.add_argument('--net', default=None, help='Net weights for predicting the boxes')
parser.add_argument('--img_path', default='./data/', help='Location to load the image to predict')
parser.add_argument('--save_path', default='./data/predicted.png', help='Location to save the predicted image')
parser.add_argument('--backbone', default="resnet50", help='Location to save training logs')

args = parser.parse_args()

model = RetinaFace(args.backbone)
model.load_state_dict(torch.load(args.net))
model = model.to(device)
model.eval()
print("[INFO] Model is loaded!")

imgs = read_image(args.img_path, None, augment=False, resize_len=[-1, -1])
init_img = get_PIL_image(imgs)

with torch.no_grad():
    cls, box = model.predict_image(imgs)
    print("Number of people predicted:", cls.shape[0])
    if cls.shape[0] > 0:
        print("Predictions:")
        for person in range(cls.shape[0]):
            print(person, "--> Conf:", cls[person], "| Box:", box[person,:])
    if args.save_path is not None:
        draw_boxes(init_img, box, save_dir=args.save_path)
    
    
