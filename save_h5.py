
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import h5py
import argparse
import random
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Subset
import pandas as pd

#datasets
from Datasets.ILSVRC import ImageNetDataset_val

#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import timm
from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224

#methods
from Methods.AGCAM.AGCAM import AGCAM
from Methods.LRP.ViT_explanation_generator import LRP
from Methods.AttentionRollout.AttentionRollout import VITAttentionRollout
from Methods.TIS.tis import TISWrapper
from Methods.ViTCX.vitcx import ViTCXWrapper
from Methods.BT.bt import BTTWrapper, BTHWrapper
from Methods.TAM.tam import TAMWrapper
from Methods.Chefer2.chefer2 import Chefer2Wrapper
from Methods.Better_AGCAM.Better_AGCAM import ScoreAGC

parser = argparse.ArgumentParser(description='save heatmaps in h5')
parser.add_argument('--method', type=str, choices=['agcam', 'rollout', 'tis', 'vitcx', 'btt', 'bth', 'tam', 'chefer1', 'chefer2', 'scoreagc'])
parser.add_argument('--save_root', type=str, required=True)
parser.add_argument('--data_root', type=str, required=True)
args = parser.parse_args()



MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
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

# root to save the h5 file
save_root = args.save_root
save_name=""

# root of the dataset
data_root = args.data_root

# transformation for ILSVRC (original)
# test_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define the unnormalize transform
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse the normalization applied to an image tensor.
    
    Args:
        tensor (torch.Tensor): A normalized image tensor (C, H, W).
        mean (list): Mean values used for normalization.
        std (list): Standard deviation values used for normalization.
    
    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape to (C, 1, 1) for broadcasting
    std = torch.tensor(std).view(3, 1, 1)    # Reshape to (C, 1, 1) for broadcasting
    return tensor * std.to('cuda') + mean.to('cuda')  # Reverse the normalization
# unnormalize = transforms.Compose([
#     transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
#     transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
# ])

validset = ImageNetDataset_val(
    root_dir=data_root,
    transforms=test_transform,
)

# create model and the method
class_num=1000
save_name +="ILSVRC"

def attn_method_model():
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location='cuda')
    model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to('cuda')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to('cuda')
    return model


model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
model = model.eval()
model = model.to('cuda')

if args.method == 'chefer1':
    state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location=device)
    model = LRP_vit_base_patch16_224(device, num_classes=class_num).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    method = LRP(model, device=device)

if args.method=="agcam":
    model = attn_method_model()
    method = AGCAM(model)
    save_name +="_agcam"
elif args.method=="chefer1":
    method = LRP(model, device=device)
    save_name+="_chefer1"
elif args.method=="rollout":
    model = attn_method_model()
    method = VITAttentionRollout(model, device=device)
    save_name+='_rollout'
elif args.method == 'tis':
    method = TISWrapper(model=model)
elif args.method == 'vitcx':
    method = ViTCXWrapper(model=model)
elif args.method == 'btt':
    method = BTTWrapper(model=model)
elif args.method == 'bth':
    method = BTHWrapper(model=model)
elif args.method == 'tam':
    method = TAMWrapper(model=model)
elif args.method == 'chefer2':
    method = Chefer2Wrapper(model=model)
elif args.method == 'scoreagc':
    model = attn_method_model()
    method = ScoreAGC(
        model, 
        plus=0, 
        vitcx_score_formula=False, 
        add_noise=True,
        score_minmax_norm=True,
        normalize_cam_heads=True,
        is_head_fuse=False,
    )

print("save the data in ", save_root)

# make h5py file
file = h5py.File(os.path.join(save_root, save_name+".hdf5"), 'w')
file.create_group('label')  # gruop to save the class label of each image
file.create_group('image')  # group to save the original image
file.create_group('cam')    # group to save the heatmap visualization

g_label=file['label']
g_image=file['image']
g_cam = file['cam']


validloader = DataLoader(
    dataset = validset,
    batch_size=1,
    shuffle = False,
)

subset_indices = pd.read_csv('/kaggle/working/better_agc_ubuntu/2000idx_ILSVRC2012.csv', header=None)[0].to_numpy()
subset = Subset(validloader.dataset, subset_indices)
subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

with torch.enable_grad():
    for data in tqdm(subset_loader):
        image = data['image'].to(device)
        label = data['label'].to(device)
        filename = data['filename']

        # generate heatmap
        prediction, heatmap = method.generate(image, label)

        # solve error "TypeError: Tensor is not a torch image." by below lines  
        if args.method in ['tis', 'btt', 'bth', 'tam', 'chefer2', 'scoreagc']: 
            heatmap = heatmap.reshape(1, 1, 14, 14) 

        # resize the heatmap
      
        resize = transforms.Resize((224, 224))
        heatmap = resize(heatmap[0])

        # normalize the heatmap from 0 to 1
        heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
        heatmap = heatmap.detach().cpu().numpy()

        #unnormalize the image to save the original input image
        image = unnormalize(image)
        image = image.detach().cpu().numpy()

        # save the class label, input image and the heatmap
        g_image.create_dataset(filename[0], data=image)
        g_label.create_dataset(filename[0], data=label.detach().cpu().numpy())
        g_cam.create_dataset(filename[0], data=heatmap)

        
        
file.close()