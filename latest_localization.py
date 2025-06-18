# ----------------- ATTENTION PLEASE FILL XAI METHOD YOU WANT TO USE ----------
# AVAILABLE METHODS: lrp, agc, better_agc, better_agc_plus1, attention rollout

import torch
import torchvision.transforms as transforms
import os
import numpy as np
import gc
import argparse
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.bounding_box import getBoudingBox_multi, box_to_seg
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
#methods    
from Methods.AGCAM.AGCAM import AGCAM
from Methods.Better_AGCAM.Better_AGCAM import BetterAGC, ScoreAGC, BetterAGC_ver2, BetterAGC_softmax, BetterAGC_cluster, BetterAGC_cluster_add_noise, ScoreAGC_no_grad, ScoreAGC_head_fusion
from Methods.Chefer2.chefer2 import Chefer2Wrapper
from Methods.TAM.tam import TAMWrapper
#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours

import timm
import torch.nn as nn

# dataset
from Datasets.ILSVRC import ImageNetDataset_val, Cub2011
from torchvision.datasets import VOCDetection

from torch.utils.data import Subset
import pandas as pd

from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224
from Methods.LRP.ViT_explanation_generator import LRP

from Methods.AttentionRollout.AttentionRollout import VITAttentionRollout
from Methods.TIS.tis import TISWrapper
from Methods.ViTCX.vitcx import ViTCXWrapper
from Methods.BT.bt import BTTWrapper, BTHWrapper

import csv
from csv import DictWriter

from torchvision.transforms import Resize

import argparse

parser = argparse.ArgumentParser(description='Generate xai maps')
parser.add_argument('--method',   type=str, default='agc',                       help='method name')
parser.add_argument('--num_heatmaps',   type=str, default=30,                       help='number of heatmaps after clustering')
parser.add_argument('--dataset',   type=str, default='imagenet',                       help='imagenet or PASCAL VOC')
parser.add_argument('--start_layer',   type=str, default=1,                       help='layer for early stopping')
args = parser.parse_args()

METHOD = args.method
DATASET = args.dataset
class csv_utils:
    def __init__(self, fileName):
        self.fileName = fileName
        self.fieldNames = ["label", "pixel_acc", "iou", "dice", "precision", "recall"]

    def writeFieldName(self):
        with open(self.fileName, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldNames)
            writer.writeheader()
            csvfile.close()

    def appendResult(self, img_name, pixel_acc, iou, dice, precision, recall):
        with open(self.fileName, "a") as csvfile:
            writer = DictWriter(csvfile, fieldnames=self.fieldNames)
            writer.writerow(
                {
                    "label": img_name,
                    "pixel_acc": pixel_acc.item(),
                    "iou": iou.item(),
                    "dice": dice.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                }
            )
            csvfile.close()

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
device = 'cuda' 
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)
IMG_SIZE=224
THRESHOLD = float(0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# AGC transform
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
# ])

# ------------------------------------ SET UP DATASET ---------------------------------------
if DATASET == 'imagenet':
    validset = ImageNetDataset_val(
        # root_dir='./ILSVRC',
        root_dir='/kaggle/input/ilsvrc/ILSVRC',
        transforms=transform,
    )

    validloader = DataLoader(
        dataset = validset,
        batch_size=1,
        shuffle = False,
    )
else:
    ds = VOCDetection(root="./data", year="2012", image_set="val", download=True, transform=transform)
    validloader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# Model Parameter provided by Timm library.
if DATASET == 'imagenet':
    class_num = 1000
else:    
    class_num=20

# name = "The localization score of attention_rollout" 
# METHOD = 'attention_rollout'
export_file = METHOD + '_results.csv'
data_file = METHOD + '_data.csv'

def load_state_dict(dataset, path_imagenet, path_custom):
    if dataset == 'imagenet':
        return model_zoo.load_url(path_imagenet, progress=True, map_location='cuda')
    else:
        return torch.load(path_custom)

def create_vit_model(model_name, num_classes, state_dict, strict=True):
    model = ViT_Ours.create_model(model_name, pretrained=True, num_classes=num_classes).to('cuda')
    model.load_state_dict(state_dict if isinstance(state_dict, dict) else state_dict['model_state'], strict=strict)
    return model.eval()

def create_timm_model(model_name='vit_base_patch16_224', num_classes=1000):
    model = timm.create_model(model_name=model_name, pretrained=True, pretrained_cfg='orig_in21k_ft_in1k', num_classes=num_classes)
    return model.eval().to('cuda')

path_imagenet = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth'
path_custom = '/kaggle/working/better_agc_ubuntu/vit_pascal_voc_60.pth'

if METHOD == 'scoreagc':
    if DATASET == 'imagenet':
        state_dict = load_state_dict(DATASET, path_imagenet, path_custom)
        model = create_vit_model(MODEL, 1000, state_dict)
    else:
        state_dict = torch.load(path_custom)
        model = create_vit_model(MODEL, 20, state_dict, strict=False)
        model.head = nn.Linear(model.head.in_features, 20).to('cuda')
    method = ScoreAGC(model)

elif METHOD in ['scoreagc_head_fusion', 'better_agc', 'better_agc_ver2', 'better_agc_softmax', 'scoreagc_no_grad', 'better_agc_cluster', 'better_agc_cluster_add_noise']:
    state_dict = load_state_dict('imagenet', path_imagenet, path_custom)
    model = create_vit_model(MODEL, class_num, state_dict)
    if METHOD == 'scoreagc_head_fusion':
        method = ScoreAGC_head_fusion(model, score_minmax_norm=True, head_fusion='mean')
    elif METHOD == 'better_agc':
        method = BetterAGC(model)
    elif METHOD == 'better_agc_ver2':
        method = BetterAGC_ver2(model)
    elif METHOD == 'better_agc_softmax':
        method = BetterAGC_softmax(model)
    elif METHOD == 'scoreagc_no_grad':
        method = ScoreAGC_no_grad(model)
    elif METHOD == 'better_agc_cluster':
        method = BetterAGC_cluster(model, num_heatmaps=int(args.num_heatmaps))
    elif METHOD == 'better_agc_cluster_add_noise':
        method = BetterAGC_cluster_add_noise(model, num_heatmaps=30)

elif METHOD == 'agc':
    state_dict = load_state_dict(DATASET, path_imagenet, path_custom)
    num_classes = 1000 if DATASET == 'imagenet' else 20
    model = create_vit_model(MODEL, num_classes, state_dict, strict=DATASET == 'imagenet')
    method = AGCAM(model)

elif METHOD == 'chefer1':
    state_dict = load_state_dict('imagenet', path_imagenet, path_custom)
    model = LRP_vit_base_patch16_224('cuda').to('cuda')
    model.load_state_dict(state_dict)
    method = LRP(model, device='cuda')

elif METHOD == 'rollout':
    state_dict = load_state_dict(DATASET, path_imagenet, path_custom)
    num_classes = 1000 if DATASET == 'imagenet' else 20
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=num_classes).to('cuda')
    if DATASET != 'imagenet':
        model.head = nn.Linear(model.head.in_features, 20).to('cuda')
    model.load_state_dict(state_dict if DATASET == 'imagenet' else state_dict['model_state'], strict=DATASET == 'imagenet')
    model.eval()
    method = VITAttentionRollout(model, device=device)

elif METHOD == 'chefer2':
    model = create_timm_model()
    method = Chefer2Wrapper(model=model)

elif METHOD == 'tam':
    model = create_timm_model()
    method = TAMWrapper(model=model)

elif METHOD == 'tis':
    model = create_timm_model()
    state_dict = model_zoo.load_url(path_imagenet, progress=True, map_location='cuda')
    model.load_state_dict(state_dict)
    method = TISWrapper(model=model)

elif METHOD == 'vitcx':
    model = create_timm_model()
    state_dict = model_zoo.load_url(path_imagenet, progress=True, map_location='cuda')
    model.load_state_dict(state_dict)
    method = ViTCXWrapper(model=model)

elif METHOD == 'btt':
    model = create_timm_model()
    method = BTTWrapper(model=model)

elif METHOD == 'bth':
    model = create_timm_model()
    method = BTHWrapper(model=model)


subset_indices = pd.read_csv('/kaggle/working/better_agc_ubuntu/2000idx_ILSVRC2012.csv', header=None)[0].to_numpy()
first_index = subset_indices[0]
last_index = subset_indices[-1]

subset = Subset(validloader.dataset, subset_indices)
subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

print(f"[CURRENT DATASET]: {DATASET}")
print(f"[XAI METHOD]: {METHOD}")

VOC_CLASSES = {
    "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
    "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
    "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
    "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
}

def normalize_mask(mask):
    return (mask - mask.min() + 1e-5) / (mask.max() - mask.min() + 1e-5)

def compute_metrics(output, target):
    eps = 1e-5
    tp = torch.sum(output * target)
    fp = torch.sum(output * (1 - target))
    fn = torch.sum((1 - output) * target)
    tn = torch.sum((1 - output) * (1 - target))
    
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    return pixel_acc, dice, precision, recall, iou

def process_image(image, label, bnd_box, method):
    prediction, saliency_map = method(image) if 'better_agc' in METHOD or METHOD == 'scoreagc' else method.generate(image)
    if prediction != label:
        return None

    if METHOD != 'vitcx':
        mask = torch.nn.Upsample(224, mode='bilinear', align_corners=False)(
            saliency_map.reshape(1, 1, 14, 14)
        )
    else:
        mask = saliency_map.unsqueeze(0).unsqueeze(0)

    mask = normalize_mask(mask)
    seg_label = box_to_seg(bnd_box.unsqueeze(0).to('cuda'))
    mask_bnd_box = getBoudingBox_multi(mask, threshold=THRESHOLD).to('cuda')
    seg_mask = box_to_seg(mask_bnd_box).to('cuda')

    output = seg_mask.view(-1)
    target = seg_label.view(-1).float()
    
    return compute_metrics(output, target)

def evaluate_imagenet():
    results = {'pixel_acc': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'iou': 0.0}
    num_img = 0
    for idx, data in enumerate(tqdm(subset_loader)):
        if num_img >= 1:
            break
        image = data['image'].to('cuda')
        label = data['label'].item()
        bnd_box = data['bnd_box'].to('cuda').squeeze(0)

        metrics = process_image(image, label, bnd_box, method)
        if metrics:
            for key, val in zip(results.keys(), metrics):
                results[key] += val
            num_img += 1
    return results, num_img

def evaluate_voc():
    results = {'pixel_acc': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'iou': 0.0}
    num_img = 0
    for idx, (image, targets) in enumerate(tqdm(validloader)):
        obj_list = targets[0]['annotation']['object']
        if len(obj_list) != 1:
            continue
        obj = obj_list[0]

        width = targets[0]['annotation']['size']['width']
        height = targets[0]['annotation']['size']['height']
        bbox = obj['bndbox']
        label = VOC_CLASSES[obj['name']]
        bnd_box = torch.tensor([
            int(int(bbox['xmin']) / width * 224),
            int(int(bbox['ymin']) / height * 224),
            int(int(bbox['xmax']) / width * 224),
            int(int(bbox['ymax']) / height * 224),
        ]).to('cuda')

        image_tensor = torch.stack(image).to('cuda')

        metrics = process_image(image_tensor, label, bnd_box, method)
        if metrics:
            for key, val in zip(results.keys(), metrics):
                results[key] += val
            num_img += 1
        if num_img == 2000:
            break
    return results, num_img

# Main execution block
with torch.enable_grad():
    if DATASET == 'imagenet':
        results, num_img = evaluate_imagenet()
    else:
        results, num_img = evaluate_voc()

    if num_img > 0:
        print(f"[INFO] Evaluated {num_img} images")
        print(f"Pixel Accuracy: {results['pixel_acc'] / num_img:.4f}")
        print(f"IoU: {results['iou'] / num_img:.4f}")
        print(f"Dice: {results['dice'] / num_img:.4f}")
        print(f"Precision: {results['precision'] / num_img:.4f}")
        print(f"Recall: {results['recall'] / num_img:.4f}")


if 'cluster' in METHOD:
    print(f'{METHOD} - num of heatmaps: {args.num_heatmaps}')
else:
    print(METHOD)

print("result==================================================================")
if DATASET != 'imagenet':
    print("Total images: ", total_counts)
print("number of images correctly predicted: ", num_img)
print("Threshold: ", THRESHOLD)
print("pixel_acc: {:.4f} ".format((pixel_acc/num_img).item()))
print("iou: {:.4f} ".format((iou/num_img).item()))
print("dice: {:.4f} ".format((dice/num_img).item()))
print("precision: {:.4f} ".format((precision/num_img).item()))
print("recall: {:.4f} ".format((recall/num_img).item()))

# --------- For visualize heatmap --------------

# print('[AFTER CLUSTERING] heatmaps shape', saliency_maps.shape)
# npz_name = args.method
        
# # saliencies_maps = torch.stack(saliency_maps) #saliency_maps.shape = [num_images, 1, 224, 224]
# np.savez(os.path.join('npz', npz_name), saliency_maps.detach().cpu().numpy())

# print('Saliency maps saved to npz.')
