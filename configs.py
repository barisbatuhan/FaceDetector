"""
###########################################################
# COMMON PARAMETERS
###########################################################
"""
# File Paths
wf_path             = "/datasets/widerface/WIDER_"
wf_labels_path      = "/datasets/widerface/retinaface/"
wf_eval_path        = "./data/"
COCO_path           = "/datasets/COCO/"
COCO_styled_path    = "/userfiles/baristopal20/COCO_styled/"
manga109_path       = "/userfiles/baristopal20/manga109/"
custom_comics_path  = "/userfiles/baristopal20/custom_comics/"
ICF_tr_path         = "/datasets/iCartoonFace2020/personai_icartoonface_dettrain/icartoonface_dettrain/"
ICF_tr_labels_path  = "/datasets/iCartoonFace2020/personai_icartoonface_dettrain/"
ICF_val_path        = "/datasets/iCartoonFace2020/personai_icartoonface_detval/"
data_dir            = "./data/"

# Device Parameters
run_gpu             = True
device              = "cuda" if run_gpu else "cpu"

# Train Parameters
weight_decay        = 0.0005

"""
###########################################################
# DETECTOR PARAMETERS
###########################################################
"""
# Training Parameters
rf_bs               = 20

# Loss Calculation Metrics
pos_iou             = 0.5
neg_iou             = 0.3
ohem_ratio          = 3 
lambda1             = 2
variances           = [0.2, 0.1]
