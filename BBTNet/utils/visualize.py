import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def draw_saliency(model, imgs, init_img):
    w, h = init_img.size
    px = 1/plt.rcParams['figure.dpi']
#     fig = plt.figure(frameon=False)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(1.5*w*px, h*px)
    
    imgs.requires_grad_()
    outs = model.fpn(model.backbone(imgs))
    prob = model.p2_pred[:-1](outs[0]) + model.p4_pred[:-1](outs[2]) + model.p6_pred[:-1](outs[4])
    prob.backward()
    
    saliency, _ = torch.max(imgs.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(init_img.size[1], init_img.size[0])
    
    # Visualize the image and the saliency map
    ax[0].imshow(init_img)
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('Prob:' + str(torch.sigmoid(prob / 3).data.item()))
    plt.show()

def draw_boxes(img, box, save_dir=None):
    w, h = img.size
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w*px, h*px)
    plt.imshow(img)
    ax = plt.gca()

    for idx in range(box.shape[0]):
        # Create a Rectangle patch
        box_len = min(box[idx,2]-box[idx,0], box[idx,3]-box[idx,1])
        rect = Rectangle(
            (box[idx,0], box[idx,1]),
            box[idx,2]-box[idx,0],
            box[idx,3]-box[idx,1],
            linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir)
    
    
def draw_different_boxes(img, box_arr, conf_arr, titles):
    w, h = img.size
    wsize = 3
    if len(box_arr) % wsize == 0: 
        hsize = (len(box_arr) // wsize) 
    else:
        hsize = (len(box_arr) // wsize + 1)
    w = (w + 100) * wsize
    h = (h + 50) * hsize
    
    px = 1/plt.rcParams['figure.dpi']
    f, axarr = plt.subplots(hsize, wsize)
    f.set_size_inches(w*px, h*px)
    
    colors = ["r"] # "b", "orange", "green", "pink", "yellow"

    for j, box in enumerate(box_arr):
        widx = j % wsize
        hidx = j // wsize
        conf = conf_arr[j]
        if hsize == 1:
            ax = axarr[j]
        else:
            ax = axarr[hidx, widx]
        ax.imshow(img)
        ax.title.set_text(titles[j])
        color = colors[j%len(colors)]
        for idx in range(box.shape[0]):
            # Create a Rectangle patch
            box_len = min(box[idx,2]-box[idx,0], box[idx,3]-box[idx,1])
            rect = Rectangle(
                (box[idx,0], box[idx,1]), 
                box[idx,2]-box[idx,0], box[idx,3]-box[idx,1],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            cx = box[idx,0]
            cy = box[idx,1] + 12
            ax.text(cx, cy, str(conf[idx])[:5])
            if hsize == 1:
                axarr[j] = ax 
            else:
                axarr[hidx, widx] = ax
    plt.show()