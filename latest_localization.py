# ----------------- ATTENTION PLEASE FILL XAI METHOD YOU WANT TO USE ----------
# AVAILABLE METHODS: lrp, agc, better_agc, better_agc_plus1, attention rollout
METHOD = 'better_agc_cluster'

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
from Methods.Better_AGCAM.Better_AGCAM import BetterAGC, BetterAGC_plus1, BetterAGC_ver2, BetterAGC_softmax, BetterAGC_cluster
#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours

import timm

# dataset
from Datasets.ILSVRC import ImageNetDataset_val

from torch.utils.data import Subset
import pandas as pd

from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224
from Methods.LRP.ViT_explanation_generator import LRP

from Methods.AttentionRollout.AttentionRollout import VITAttentionRollout

import csv
from csv import DictWriter

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

validset = ImageNetDataset_val(
    # root_dir='./ILSVRC',
    root_dir='/kaggle/input/ilsvrc/ILSVRC',
    transforms=transform,
)

# Model Parameter provided by Timm library.
class_num=1000

# name = "The localization score of attention_rollout" 
# METHOD = 'attention_rollout'
export_file = METHOD + '_results.csv'
data_file = METHOD + '_sigmoid_data.csv'

if METHOD == 'better_agc_plus1':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC_plus1(model)
elif METHOD == 'better_agc':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC(model)
elif METHOD == 'better_agc_ver2':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC_ver2(model)
elif METHOD == 'better_agc_softmax':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC_softmax(model)
elif METHOD == 'agc':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = AGCAM(model)
elif METHOD == 'better_agc_cluster':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC_cluster(model, thresold=0.9)
elif METHOD == 'lrp':
    model = LRP_vit_base_patch16_224('cuda', num_classes=1000).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = LRP(model, device='cuda')
elif METHOD == 'attention rollout':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = VITAttentionRollout(model, device=device)

print(f"[XAI METHOD]: {METHOD}")

validloader = DataLoader(
    dataset = validset,
    batch_size=1,
    shuffle = False,
)
subset_indices = pd.read_csv('/kaggle/working/better_agc_ubuntu/2000idx_ILSVRC2012.csv', header=None)[0].to_numpy()
subset = Subset(validloader.dataset, subset_indices)
subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

with torch.enable_grad():      
    idx = 0
    data_idx = -1
    num_img = 0
    pixel_acc = 0.0
    dice = 0.0
    precision = 0.0
    recall = 0.0
    iou = 0.0
    
    fieldnames = ['num_img', 'pixel_acc', 'iou', 'dice', 'precision', 'recall']
    fieldnames_data = ['idx', 'num_img', 'pixel_acc', 'iou', 'dice', 'precision', 'recall']
    csvUtils = csv_utils(export_file)
    csvUtils.writeFieldName()

    for data in tqdm(subset_loader):
        image = data['image'].to('cuda')
        label = data['label'].to('cuda')
        bnd_box = data['bnd_box'].to('cuda').squeeze(0)

        if 'better_agc' in METHOD:
            prediction, saliency_map = method(image)
        else:
            prediction, saliency_map = method.generate(image)
        # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
        if prediction!=label:
            continue
        
        mask = saliency_map.reshape(1, 1, 14, 14)
            
        # Reshape the mask to have the same size with the original input image (224 x 224)
        upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
        mask = upsample(mask)

        # Normalize the heatmap from 0 to 1
        mask = (mask-mask.min() + 1e-5)/(mask.max()-mask.min() + 1e-5)

        # To avoid the overlapping problem of the bounding box labels, we generate a 0-1 segmentation mask from the bounding box label.
        seg_label = box_to_seg(bnd_box).to('cuda')

        # From the generated heatmap, we generate a bounding box and then convert it to a segmentation mask to compare with the bounding box label.
        
        mask_bnd_box = getBoudingBox_multi(mask, threshold=THRESHOLD).to('cuda')
        seg_mask = box_to_seg(mask_bnd_box).to('cuda')
        
        output = seg_mask.view(-1, )
        target = seg_label.view(-1, ).float()

        tp = torch.sum(output * target)  # True Positive
        fp = torch.sum(output * (1 - target))  # False Positive
        fn = torch.sum((1 - output) * target)  # False Negative
        tn = torch.sum((1 - output) * (1 - target))  # True Negative
        eps = 1e-5
        pixel_acc_ = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice_ = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision_ = (tp + eps) / (tp + fp + eps)
        recall_ = (tp + eps) / (tp + fn + eps)
        iou_ = (tp + eps) / (tp + fp + fn + eps)
        
        pixel_acc += pixel_acc_
        dice += dice_
        precision += precision_
        recall += recall_
        iou += iou_
        num_img+=1
    
        csvUtils.appendResult(
            data["filename"][0], pixel_acc_, iou_, dice_, precision_, recall_
        )

print(METHOD)
print("result==================================================================")
print("number of images: ", num_img)
print("Threshold: ", THRESHOLD)
print("pixel_acc: {:.4f} ".format((pixel_acc/num_img).item()))
print("iou: {:.4f} ".format((iou/num_img).item()))
print("dice: {:.4f} ".format((dice/num_img).item()))
print("precision: {:.4f} ".format((precision/num_img).item()))
print("recall: {:.4f} ".format((recall/num_img).item()))