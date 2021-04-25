import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

from configs import run_gpu


def read_image(img_path, boxes, augment=True, resize_len=[640, 640]):
    img = Image.open(img_path).convert('RGB')
    box = np.array(boxes).copy() if boxes is not None else None
    
    if augment:
        img, box = crop(img, box)
        img = distort_color(img)
        img, box = horizontal_flip(img, box)
        img, box = resize(img, box, resize_len)
    
    elif resize_len[0] > 0 and resize_len[0] == resize_len[1]:
        img = squaritize(img)
        img, box = resize(img, box, resize_len)
    else:
        img, box = resize(img, box, resize_len)
    
    img = normalize(transforms.ToTensor()(img))
    box = torch.Tensor(box) if box is not None else None
    if run_gpu:
        img = img.cuda(); box = box.cuda() if box is not None else None
    
    if box is not None:
        return img, box
    else:
        return img


def normalize(img, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    return TF.normalize(img, mean=means, std=stds)

"""
mode -> 0 : add right and bottom size
mode -> 1 : add left and upper side
mode -> 2 : center the image
"""
def squaritize(img, mode=0):
    W, H = img.size
    if W == H:
        return img
    max_len = max(W, H)
    delta_w = max_len - W; delta_h = max_len - H
    if mode == 0:
        padding = (0, 0, delta_w, delta_h)
    elif mode == 1:
        padding = (delta_w, delta_h, 0, 0)
    else:
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(img, padding)


def distort_color(img):
    if np.random.rand() > 0.5:
        br_strength = np.random.randint(4, 21) / 10
        img = TF.adjust_brightness(img, br_strength) # 0.4 - 2.0 range
    if np.random.rand() > 0.5:
        con_strength = np.random.randint(4, 31) / 10
        img = TF.adjust_contrast(img, con_strength) # 0.4 - 3.0 range
    if np.random.rand() > 0.5:
        hue_strength = np.random.randint(-5, 5) / 10
        img = TF.adjust_hue(img, hue_strength) # -0.4 - 0.4 range 
    if np.random.rand() > 0.5:
        sat_strength = np.random.randint(-1, 20) / 10
        img = TF.adjust_saturation(img, sat_strength) # 0.0 - 2.0 range
    return img


def horizontal_flip(img, box):
    if np.random.rand() > 0.5:
        W, H = img.size
        img = TF.hflip(img)
        if box is not None:
            box[:,0:4:2] = W - box[:,0:4:2]
            box[np.where(box > W)] = -1
            temp = box[:,0].copy()
            box[:,0] = box[:,2]
            box[:,2] = temp
    return img, box


def resize(img, box, resize_len):
    W, H = img.size
    resize_len[0] = W if resize_len[0] < 0 else resize_len[0]
    resize_len[1] = H if resize_len[1] < 0 else resize_len[1]
    w_ratio = resize_len[0]/W; h_ratio = resize_len[1]/H

    img = TF.resize(img, (resize_len[1], resize_len[0]))
    if box is not None:
        box[:,0:4:2] *= w_ratio
        box[:,1:4:2] *= h_ratio
    return img, box


def crop(img, box):
    W, H = img.size
    min_len = min(W, H)
    crop_ratios = [0.3, 0.45, 0.6, 0.8, 1.0]
    
    while True: # at least center of one face will be in cropped image
        crop_ratio = np.random.choice(crop_ratios)
        side_len = min_len * crop_ratio - 1
        w_start = np.random.randint(0, max(1, W - side_len + 1))
        h_start = np.random.randint(0, max(1, H - side_len + 1))

        centers = box[:,2:4] - box[:,0:2]
        in_bound = (w_start <= centers[:,0])*(centers[:,0] <= w_start + side_len)
        in_bound *= (h_start <= centers[:,1])*(centers[:,1] <= h_start + side_len)
        in_bound = np.where(in_bound == True)[0].flatten()

        if len(in_bound) > 0:
            break

    box = box[in_bound, :]

    img = TF.crop(img, h_start, w_start, side_len, side_len)
    if box is not None:
        box[:,0:4:2] -= w_start
        box[:,1:4:2] -= h_start
        box[:,0:4] = np.minimum(side_len, np.maximum(0, box[:,0:4]))
        corr_x = np.where(box[:,2]-box[:,0] > 0)
        corr_y = np.where(box[:,3]-box[:,1] > 0)
        visible_faces = np.intersect1d(corr_x, corr_y)
        box = box[visible_faces,:]
    return img, box

def get_PIL_image(img_tensor):
    img = img_tensor.to("cpu")
    means = torch.Tensor([[[0.485]], [[0.456]], [[0.406]]])
    stds = torch.Tensor([[[0.229]], [[0.224]], [[0.225]]])
    return transforms.ToPILImage()(img * stds + means)
