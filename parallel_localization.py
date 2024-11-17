
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

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True, max_split_size_mb:512"

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
        fieldnames = ['num_img', 'pixel_acc', 'iou', 'dice', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
        for data in tqdm(validloader):
            if num_img == 1:
                break
            image = data['image'].to(device)
            # image = data['image']
            label = data['label'].to(device)
            # label = data['label']
            bnd_box = data['bnd_box'].squeeze(0).to(device)
            # bnd_box = data['bnd_box'].squeeze(0)
            

            if (args.method == 'better_agcam'):
                # model = model.cuda()
                prediction, better_agc_heatmap, output_truth = method.generate(image)
                # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
                if prediction!=label:
                    continue
                # model = model.cpu()
                # output_truth = output_truth.cpu()
                # prediction = prediction.cpu()
                transformed_img = image[0]
                
                agc_scores = []
                
                first_output_mask = None
                loop_m = None
                loop_heatmap = None
                cam_head_dau_tien = None
                cam_head_dau_tien_resized = None
                agc_scores_loop = None
                loop_head_cams_resized = []
                output_mask_loop = []
                ms = []
                cams_loop = []
                cams_after_generate = []
                cams_after_resized = []
                # for i in range(better_agc_heatmap.size(1)):     # Loop over the first dimension (12)
                #     for j in range(better_agc_heatmap.size(2)): # Loop over the second dimension (12)
                #         # print(better_agc_heatmap[0][i][j].shape)
                #         cam_head_dau_tien = better_agc_heatmap[0][i][j]
                #         cams_after_generate.append(better_agc_heatmap[0][i][j])
                #         tensor_heatmap = transforms.Resize((224, 224))(better_agc_heatmap[0][i][j])
                #         cams_after_resized.append(tensor_heatmap)
                #         cam_head_dau_tien_resized = tensor_heatmap
                #         # print(tensor_heatmap.shape)
                #         tensor_heatmap = (tensor_heatmap - tensor_heatmap.min())/(tensor_heatmap.max()-tensor_heatmap.min() + 0.0000000000001)
                #         # tensor_heatmap = tensor_heatmap.unsqueeze(0).to(device)
                #         tensor_heatmap = tensor_heatmap.unsqueeze(0).cpu()

                #         cams_loop.append(tensor_heatmap)

                #         loop_heatmap = tensor_heatmap
                        
                #         # tensor_img = transformed_img.unsqueeze(0).to(device)
                #         tensor_img = transformed_img.unsqueeze(0).cpu()
                #         print('loop: ', 'img', tensor_img.shape, 'mask', tensor_heatmap.shape)
                #         # model.zero_grad() # Niên: mình forward pass nên không cần zero_grad. Chỉ cần khi mà có loss.backward
                #         m = torch.mul( tensor_img, tensor_heatmap)
                #         ms.append(m)
                #         print('loop: masked_image', m.shape)
                        
                #         loop_m = m #!

                #         with torch.no_grad():
                #           output_mask = model(m)
                        
                #         output_mask_loop.append(output_mask)

                #         print('loop: output masked shape', output_mask.shape)
                #         first_output_mask = output_mask #!

                        
                #         agc_score = output_mask[0, prediction.item()] - output_truth[0, prediction.item()]
                #         agc_scores.append(agc_score.detach().cpu().numpy())

                # agc_scores_loop = agc_scores 
                    
                    

                # -------------------------- CUDA ------------------------------
                tensor_heatmaps = better_agc_heatmap[0]
                print('dau tien', tensor_heatmaps.shape)
                # tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps[0])
                tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
                tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
                # tensor_heatmaps = F.interpolate(tensor_heatmaps, size=(244, 244), mode='bilinear', align_corners=False)
                print('resized', tensor_heatmaps.shape)
                # tensor_heatmaps = tensor_heatmaps.cpu()

                # Compute min and max along each image
                min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height

                # Normalize using min-max scaling
                tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero


                tensor_img = transformed_img.unsqueeze(0)
                print('input image', tensor_img.shape)
                m = torch.mul(tensor_heatmaps, tensor_img)
                print('mask shape: ', m.shape)
                
                with torch.no_grad():
                    output_mask = model(m)
                print('output maksed', output_mask.shape)
                # output_mask = output_mask.cpu()

                agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
                print('score shape: ', agc_scores.shape)
                agc_scores = agc_scores.detach().cpu().numpy()

                masks = better_agc_heatmap[0].cpu()

                e_x = np.exp(agc_scores - np.max(agc_scores)) 
                agc_scores = e_x / e_x.sum(axis=0)
                agc_scores = agc_scores.reshape(masks.shape[0], masks.shape[1])
                
                my_cam = (agc_scores[:, :, None, None, None] * masks.detach().cpu().numpy()).sum(axis=(0, 1))
                
                mask = torch.from_numpy(my_cam)
                mask = mask.unsqueeze(0)

                # ---------------- CPU ---------------------
                # tensor_heatmaps = better_agc_heatmap[0].detach().clone()
                # print('dau tien', tensor_heatmaps.shape)
                # # tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps[0])
                # print('Kiem tra cac CAM cua head sau khi generate', torch.sum(tensor_heatmaps))
                # print('Kiem tra cac CAM cua head sau khi generate trong loop', torch.sum(torch.stack(cams_after_generate)))
                # tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
                # print('Kiem tra CAM sau khi reshape trong parallel', torch.sum(tensor_heatmaps))
                # print('Kiem tra CAM thu 13 trong parallel va loop', tensor_heatmaps[12] == cams_after_generate[12])
                # print('Kiem tra CAM dau tien trong parallel va loop', tensor_heatmaps[0] == cams_after_generate[0])
                # # print('check shape cua cam loop', cam_head_dau_tien.shape)
                # # print('check shape cua cam parallel', tensor_heatmaps[0].shape)
                # # print('check cam cua head dau tien', 'giong' if not False in (tensor_heatmaps[0] == cam_head_dau_tien) else 'khong giong')
                # tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
                # print('Kiem tra CAM sau khi resized', tensor_heatmaps[0] == torch.stack(cams_after_resized)[1])
                # # print('check shape cua cam loop sau khi resize', cam_head_dau_tien_resized.shape)
                # # print('check shape cua cam parallel sau khi resize', tensor_heatmaps[0].shape)
                # # print('check cam cua head dau tien sau khi resize', 'giong' if not False in (tensor_heatmaps[0] == cam_head_dau_tien_resized) else 'khong giong')
                
                # # tensor_heatmaps = (tensor_heatmap - tensor_heatmap.min())/(tensor_heatmap.max()-tensor_heatmap.min() + 0.0000000000001)

                # # Compute min and max along each image
                # min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                # max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height

                # # Normalize using min-max scaling
                # tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero


                # # tensor_heatmaps = F.interpolate(tensor_heatmaps, size=(244, 244), mode='bilinear', align_corners=False)
                # print('parallel: mask shape', tensor_heatmaps.shape)
                # print('kiem tra truoc khi nhan trong loop', torch.sum(torch.stack(cams_loop)))
                # print('kiem tra truoc khi nhan', torch.sum(tensor_heatmaps))
                # tensor_heatmaps = tensor_heatmaps.cpu()
                # tensor_img = transformed_img.unsqueeze(0).cpu()
                
                # m = torch.mul(tensor_heatmaps, tensor_img)
                # # print('check shape cua masked image trong loop', loop_m[0].shape)
                # # print('check shape cua masked image trong parallel', m[0].shape)
                # # print('check masked image cua loop va parallel co giong nhau khong? ', 'giong' if not False in (m[0] == loop_m[0]) else 'khong giong')
                # # print('co ton tai hay khong? ' 'co ton tai' if loop_m[0] in m else 'khong ton tai') 
                # print('kiem tra sau khi nhan trong loop: ', torch.sum(torch.stack(ms)))
                # print('Kiem tra sau khi nhan', torch.sum(m))
                # with torch.no_grad():
                #     output_mask = model(m)
                # # print('parallel: output_masked shape: ', output_mask.shape)
                # # print('check shape', output_mask[0].shape, first_output_mask[0].shape)
                # # print('check tensor', output_mask[0] == first_output_mask[0])
                
                # # print('output maksed', output_mask[0].shape)
                
                # print('Kiem tra output mask (ket qua co duoc sau khi dua mask vao model): ', torch.sum(output_mask))
                # print('Kiem tra output mask (ket qua co duoc sau khi dua mask vao model trong loop): ', torch.sum(torch.stack(output_mask_loop)))
                # output_mask = output_mask.cpu()

                # print('ket qua sau khi 144 cam cua head qua model', output_mask.shape)
                # print('ket qua sau khi dua input image qua model', output_truth.shape)
                # print('ket qua tu hinh anh ban dau theo du doan: ', output_truth[0, prediction.item()])
                # print('ket qua tu 10 cam dau cua head theo du doan: ', output_mask[:10, prediction.item()])
                # agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
                # print('ket qua trung binh tu cac cam cua head sau khi tru: ', torch.mean(agc_scores))
                # # print('score loop shape', len(agc_scores_loop))
                # # print('score shape: ', agc_scores.shape)
                # agc_scores = agc_scores.detach().cpu().numpy()
                # print(agc_scores == agc_scores_loop)
                # is_exist = False
                # for score in agc_scores_loop:
                #     if not score in agc_scores:
                #         is_exist = True
                #         break
                # print('Diem khong ton tai?', is_exist)

                masks = better_agc_heatmap[0].cpu()

                e_x = np.exp(agc_scores - np.max(agc_scores)) 
                agc_scores = e_x / e_x.sum(axis=0)
                agc_scores = agc_scores.reshape(masks.shape[0], masks.shape[1])
                
                my_cam = (agc_scores[:, :, None, None, None] * masks.detach().cpu().numpy()).sum(axis=(0, 1))
                
                mask = torch.from_numpy(my_cam)
                mask = mask.unsqueeze(0)
                # --------------------------- CPU --------------------------

                
            else:
                prediction, mask = method.generate(image)
                mask = mask.reshape(1, 1, 14, 14)
                # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
                if prediction!=label:
                    continue
            
            
            
            # print(mask.shape)

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
            
            pixel_acc += pixel_acc_
            dice += dice_
            precision += precision_
            recall += recall_
            iou += iou_
            num_img+=1
        
            writer.writerow({'num_img': num_img - 1, 
                            'pixel_acc': (pixel_acc/num_img).item(),
                            'iou': (iou/num_img).item(),
                            'dice': (dice/num_img).item(),
                            'precision': (precision/num_img).item(),
                            'recall': (recall/num_img).item()})
  

print(name)
print("result==================================================================")
print("number of images: ", num_img)
print("Threshold: ", THRESHOLD)
print("pixel_acc: {:.4f} ".format((pixel_acc/num_img).item()))
print("iou: {:.4f} ".format((iou/num_img).item()))
print("dice: {:.4f} ".format((dice/num_img).item()))
print("precision: {:.4f} ".format((precision/num_img).item()))
print("recall: {:.4f} ".format((recall/num_img).item()))
