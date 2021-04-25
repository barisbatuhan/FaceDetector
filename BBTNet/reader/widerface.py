import numpy as np
import torch
from BBTNet.reader.image_reader import read_image

from configs import device

class WF_Data:

    def __init__(self, wf_path, labels_path, batch_size=1, augment=True, img_sizes=[640, 640]):
        
        self.imgs_path = []
        self.boxes  = {}
        self.augment = augment
        self.img_sizes = img_sizes
        self.batch_size = batch_size
        
        f = open(labels_path,'r')
        lines = f.readlines()
        f.close()

        img_boxes = []; last_path = None
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if last_path is not None:
                    self.boxes[last_path] = np.array(img_boxes)
                    img_boxes = []
                path = line[2:]
                path = wf_path + 'images/' + path
                last_path = path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                labels = [float(x) for x in line]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                person_annot[2:4] += person_annot[0:2]            
                img_boxes.append(person_annot)

        self.boxes[last_path] = np.array(img_boxes)      

        if self.augment:
            self.state = np.random.permutation(len(self.imgs_path))
        else:
            self.state = np.arange(len(self.imgs_path))


    def forward(self, restart=False):
        if restart:
            if self.augment:
                self.state = np.random.permutation(len(self.imgs_path))
            else:
                self.state = np.arange(len(self.imgs_path))
        
        if self.state is None or len(self.state) < self.batch_size:
            return None, None
        else:
            indices = self.state[0:self.batch_size].copy()
            self.state = None if len(self.state)  == self.batch_size else self.state[self.batch_size:]
            
            if self.batch_size > 1 and self.img_sizes[0] > 0:
                x = torch.zeros(self.batch_size, 3, self.img_sizes[1], self.img_sizes[0]).to(device)
            
            y = []
            for i, idx in enumerate(indices):
                img_path = self.imgs_path[idx]
                boxes = self.boxes[img_path]
                img, box = read_image(img_path, boxes, self.augment, self.img_sizes)
                y.append(box)
                if self.batch_size == 1:
                    return img.unsqueeze(0), y
                else:
                    x[i,:,:,:] = img
            
            return x, y
