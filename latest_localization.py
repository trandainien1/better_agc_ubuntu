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

# dataset
from Datasets.ILSVRC import ImageNetDataset_val

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
# parser.add_argument('--npz_checkpoint',   type=str, default='',                       help='folder path storing heatmaps')
# parser.add_argument('--load_prediction',   type=str, default='true',                       help='load predictions of ViT')
args = parser.parse_args()

METHOD = args.method

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

if METHOD == 'scoreagc':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    # test scoreAGC with PCB, no binary
    method = ScoreAGC(
        model, 
        plus=0, 
        vitcx_score_formula=False, 
        add_noise=False, # ! test new scoreagc with binary cams of heads
        score_minmax_norm=True,
        normalize_cam_heads=True,
        is_head_fuse=False,
        is_binarize_cam_of_heads=True,
        handle_pixel_coverage_bias=True,
    )
if METHOD == 'scoreagc_head_fusion':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = ScoreAGC_head_fusion(model, score_minmax_norm=True, head_fusion='mean')
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
elif METHOD == 'scoreagc_no_grad':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = ScoreAGC_no_grad(model)
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
    method = BetterAGC_cluster(model, num_heatmaps=int(args.num_heatmaps))
elif METHOD == 'better_agc_cluster_add_noise':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = BetterAGC_cluster_add_noise(model, num_heatmaps=30)
elif METHOD == 'chefer1':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
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
elif METHOD == 'chefer2':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = Chefer2Wrapper(model=model)
elif METHOD == 'tam':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = TAMWrapper(model=model)
elif METHOD == 'tis':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = TISWrapper(model=model)
elif METHOD == 'vitcx':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = ViTCXWrapper(model=model)
elif METHOD == 'btt':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = BTTWrapper(model=model)
elif METHOD == 'bth':
    model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
    model = model.eval()
    method = BTHWrapper(model=model)
    
model = model.to('cuda')

print(f"[XAI METHOD]: {METHOD} - binary + PCB")

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

    for idx, data in enumerate(tqdm(subset_loader)):
        image = data['image'].to('cuda')
        label = data['label']
        bnd_box = data['bnd_box'].to('cuda').squeeze(0)
        
        if 'better_agc' in METHOD or METHOD == 'scoreagc':
            prediction, saliency_map = method(image) 
        else:
            prediction, saliency_map = method.generate(image) # [1, 1, 14, 14]

        # print('[DEBUG] PREDICTION', prediction)
        # print('[DEBUG] LABEL', label.item())
        if prediction!=label.item():
            continue
        # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
        if METHOD != 'vitcx':
            mask = saliency_map.reshape(1, 1, 14, 14) 
        
            # Reshape the mask to have the same size with the original input image (224 x 224)
            upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
        
            mask = upsample(mask)

        else:
            mask = saliency_map.unsqueeze(0).unsqueeze(0)
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

        # break # --------------- for visualize heatmaps


if 'cluster' in METHOD:
    print(f'{METHOD} - num of heatmaps: {args.num_heatmaps}')
else:
    print(METHOD)

print("result==================================================================")
print("number of images: ", num_img)
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
