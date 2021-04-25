import numpy as np
import csv
from collections import deque
import sys
sys.path.append("/kuacc/users/baristopal20/retina_pytorch/evaluate/eval_others/")

from cython_files.bbox import bbox_overlaps_cython as compute_overlap

label_map = {"face":1}
class_len = len(label_map)
new_map = {v: k for k, v in label_map.items()}
new_map = {1:'face'}

def read_csv_file(csv_path, is_dets=False):
    img_path2boxes_labels = {}
    with open(csv_path, "r") as f:
        content = csv.reader(f)
        for line in content:
            if line[0] not in img_path2boxes_labels:
                img_path2boxes_labels[line[0]] = []
            box = list(map(float, line[1:5]))
            label = line[5]
            if is_dets:  # label,scores
                assert len(line) >= 7
                score = line[-1]
                img_path2boxes_labels[line[0]].append([box, label, score])
            else:
                # 真实标签没有score,只有label
                img_path2boxes_labels[line[0]].append([box, label])
    img_paths = [path for path in img_path2boxes_labels]
    # all_label = deque()
    all_label = {}
    for img in img_path2boxes_labels:  # 得到图片名称
        all_annos = [[] * 1 for i in range(class_len)]
        pre_img_data = img_path2boxes_labels[img]  # 得到单张图片的所有数据（bbox）
        for one_data in pre_img_data:  # 得到一张图片上单个bbox
            ind = label_map[one_data[1]]
            if is_dets:
                all_annos[ind - 1].append(np.array(one_data[0]+[float(one_data[2])]))
            else:
                all_annos[ind - 1].append(np.array(one_data[0]))

        all_annos = [np.array(all_annos[j]) if len(all_annos[j]) != 0 else np.array([]) for j in range(class_len)]
        # all_label.append(all_annos)
        if(img not in all_label):
            all_label[img] = all_annos
    if is_dets:
        return all_label
    else:
        return img_paths, all_label, img_path2boxes_labels

def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calcul_map(dets, annos, save_csv=None):
    aps = {}
    for j in range(len(label_map)):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        iou_thresh = 0.5
        ctr = 1
        for key in annos.keys():
            # print("Calculating:", ctr, "/", len(annos.keys()), "-->", key)
            ctr += 1
            if(key not in dets.keys()):
                detections = [np.array([])][j]
            else:
                detections = dets[key][j]
            annotations = annos[key][j]
            num_annotations += annotations.shape[0]
            detected_annotations = deque()

            if annotations.shape[0] == 0:
                continue
            for d in detections:  # i=840
                scores = np.append(scores, d[4])
                overlaps = compute_overlap(np.expand_dims(d, axis=0).astype(np.float), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                if max_overlap >= iou_thresh and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)
        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        ap = voc_ap(recall, precision)
        aps[new_map[j + 1]] = ap
        
    return float(aps['face'])


if __name__ == "__main__":
    csv_det_path = sys.argv[1] 
    csv_anno_path = sys.argv[2]
    print("[INFO] Reading the ground truth CSV")
    img_path_list, annos, img_path2boxes_labels = read_csv_file(csv_anno_path, is_dets=False)
    print("[INFO] Reading the prediction CSV")
    dets = read_csv_file(csv_det_path, is_dets=True)
    print("[INFO] Calculating the score...")
    score = calcul_map(dets, annos)
    print(csv_det_path, "--->", score)

