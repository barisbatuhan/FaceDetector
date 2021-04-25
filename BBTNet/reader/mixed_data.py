import numpy as np
import torch
from BBTNet.reader.image_reader import read_image

from configs import device

class Mixed_Data:

    def __init__(self, wf_path, wf_labels_path, icf_path, icf_labels_path, batch_size=1, augment=True, img_sizes=[640, 640]):
        
        self.augment = augment
        self.img_sizes = img_sizes
        self.batch_size = batch_size
        
        self.boxes = {}; self.imgs_path = []
        icf_paths, icf_boxes = self.get_icf_data(icf_path, icf_labels_path)
        self.boxes.update(icf_boxes);  self.imgs_path.extend(icf_paths)
        wf_paths, wf_boxes = self.get_wf_data(wf_path, wf_labels_path)
        self.boxes.update(wf_boxes); self.imgs_path.extend(wf_paths)

        if self.augment:
            self.state = np.random.permutation(len(self.imgs_path))
        else:
            self.state = np.arange(len(self.imgs_path))


    def get_wf_data(self, wf_path, wf_labels_path):
        f = open(wf_labels_path,'r')
        lines = f.readlines()
        f.close()
        
        boxes = {}; imgs_path = []
        img_boxes = []; last_path = None
        
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if last_path is not None:
                    boxes[last_path] = np.array(img_boxes)
                    img_boxes = []
                path = line[2:]
                path = wf_path + 'images/' + path
                last_path = path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                labels = [float(x) for x in line]
                person_annot = np.zeros(4)
                person_annot[0:4] = labels[0:4]
                person_annot[2:4] += person_annot[0:2]            
                img_boxes.append(person_annot)

        boxes[last_path] = np.array(img_boxes)  
        return imgs_path, boxes
    
    def get_icf_data(self, icf_path, icf_labels_path):
        f = open(icf_labels_path,'r')
        lines = f.readlines()
        f.close()
        boxes = {}
        for line in lines:
            line = line.rstrip().split(',')
            person_annot = [float(x) for x in line[1:5]]
            if icf_path + line[0] not in self.boxes:
                boxes[icf_path + line[0]] = []
            boxes[icf_path + line[0]].append(person_annot)
        
        imgs_path = [*boxes.keys()]   
        return imgs_path, boxes
    
    
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
