import numpy as np
import torch
from BBTNet.reader.image_reader import read_image

from configs import device

class ICF_Data:

    def __init__(self, icf_path, labels_path, batch_size=1, augment=True, img_sizes=[640, 640]):
        
        self.imgs_path = []
        self.boxes  = {}
        self.augment = augment
        self.img_sizes = img_sizes
        self.batch_size = batch_size
        
        f = open(labels_path,'r')
        lines = f.readlines()
        f.close()

        for line in lines:
            line = line.rstrip().split(',')
            labels = [float(x) for x in line[1:]]
            person_annot = np.zeros(4)
            person_annot[0:4] = labels[0:4]
            if icf_path + line[0] not in self.boxes:
                self.boxes[icf_path + line[0]] = []
            self.boxes[icf_path + line[0]].append(person_annot)
        
        for k in self.boxes.keys():
            self.boxes[k] = np.array(self.boxes[k])
        
        self.imgs_path = [*self.boxes.keys()]

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
