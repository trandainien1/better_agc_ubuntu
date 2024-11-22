
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import gc
import argparse
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.bounding_box import getBoudingBox_multi, box_to_seg
import torch.utils.model_zoo as model_zoo

#datasets
from Datasets.ILSVRC import ImageNetDataset_val

#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import timm
from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224

#methods
from Methods.AGCAM.AGCAM import AGCAM
from Methods.Better_AGCAM.Better_AGCAM import Better_AGCAM
from Methods.LRP.ViT_explanation_generator import LRP
from Methods.AttentionRollout.AttentionRollout import VITAttentionRollout

import csv

import os

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['agcam', 'lrp', 'rollout', 'better_agcam'])
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--threshold', type=str, default='0.5')
args = parser.parse_args()

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
THRESHOLD = float(args.threshold)



# Image transform for ImageNet ILSVRC
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

validset = ImageNetDataset_val(
    root_dir=args.data_root,
    transforms=transform,
)

# Model Parameter provided by Timm library.
state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location=device)
class_num=1000



if args.method=="agcam":
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = AGCAM(model)
elif args.method=="better_agcam":
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    method = Better_AGCAM(model)
elif args.method=="lrp":
    model = LRP_vit_base_patch16_224(device, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = LRP(model, device=device)
elif args.method=="rollout":
    model = timm.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = VITAttentionRollout(model, device=device)

name = "The localization score of " + args.method 



validloader = DataLoader(
    dataset = validset,
    batch_size=1,
    shuffle = False,
)


with torch.enable_grad():        
    num_img = 0
    pixel_acc = 0.0
    dice = 0.0
    precision = 0.0
    recall = 0.0
    iou = 0.0
    
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['num_img', 'pixel_acc', 'iou', 'dice', 'precision', 'recall', 'img_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for data in tqdm(validloader):
            if num_img == 1000:
                break
            image = data['image'].to(device)
            label = data['label'].to(device)
            bnd_box = data['bnd_box'].squeeze(0).to(device)

            if (args.method == 'better_agcam'):
                prediction, better_agc_heatmap, output_truth = method.generate(image)
                # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
                if prediction!=label:
                    continue
              
                transformed_img = image[0]
                tensor_img = transformed_img.unsqueeze(0)
              
                tensor_heatmaps = better_agc_heatmap[0]
                tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
                tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
               
                # Compute min and max along each image
                min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
                # Normalize using min-max scaling
                tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
              
                m = torch.mul(tensor_heatmaps, tensor_img)
                
                with torch.no_grad():
                    output_mask = model(m)

                heatmaps = better_agc_heatmap[0]
              
                agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
                agc_scores = torch.sigmoid(agc_scores)
                agc_scores = agc_scores.reshape(heatmaps.shape[0], heatmaps.shape[1])
                print('agc_score shape: ', agc_scores.shape)
                print('heatmaps shape: ', heatmaps.shape)
                my_cam = (agc_scores.view(12, 12, 1, 1, 1) * heatmaps).sum(axis=(0, 1))
                
                mask = my_cam
                mask = mask.unsqueeze(0)
                
            else:
                prediction, mask = method.generate(image)
                mask = mask.reshape(1, 1, 14, 14)
                # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
                if prediction!=label:
                    continue
            
            # Reshape the mask to have the same size with the original input image (224 x 224)
            upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
            mask = upsample(mask)

            # Normalize the heatmap from 0 to 1
            mask = (mask-mask.min())/(mask.max()-mask.min())

            # To avoid the overlapping problem of the bounding box labels, we generate a 0-1 segmentation mask from the bounding box label.
            # seg_label = box_to_seg(bnd_box).to(device)
            seg_label = box_to_seg(bnd_box)


            # From the generated heatmap, we generate a bounding box and then convert it to a segmentation mask to compare with the bounding box label.
            
            # mask_bnd_box = getBoudingBox_multi(mask, threshold=THRESHOLD).to(device)
            mask_bnd_box = getBoudingBox_multi(mask, threshold=THRESHOLD)
            # seg_mask = box_to_seg(mask_bnd_box).to(device)
            seg_mask = box_to_seg(mask_bnd_box)
            
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
            
            num_img+=1
        
            writer.writerow({'num_img': num_img, 
                                'pixel_acc': (pixel_acc_).item(),
                                'iou': (iou_).item(),
                                'dice': (dice_).item(),
                                'precision': (precision_).item(),
                                'recall': (recall_).item(),
                                'img_name': data['filename'][0],
                                })
