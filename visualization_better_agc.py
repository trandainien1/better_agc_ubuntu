#If you want to generate heatmap for other image, change IMAGE_PATH

import torch
import torchvision.transforms as transforms
import numpy as np
import PIL
import matplotlib.pyplot as plt
from Datasets.ILSVRC_classes import classes
import torch.utils.model_zoo as model_zoo
import random

#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import timm

#methods
from Methods.AGCAM.AGCAM import AGCAM
from Methods.Better_AGCAM.Better_AGCAM import Better_AGCAM

import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

MODEL = 'vit_base_patch16_224'
IMG_SIZE=224
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import gc
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(777)
if device == 'cuda':
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)

#Select Image which you want to generate a heatmap for.
path_list = [
    # "samples/ILSVRC2012_val_00000125.jpeg",
    # "samples/ILSVRC2012_val_00001372.jpeg",
    # "samples/ILSVRC2012_val_00001854.jpeg",
    # "samples/catdog.png",
    # "samples/dogbird.png"
    '/content/ILSVRC2012_val_00029430.JPEG'
]
IMAGE_PATH = path_list[0]

# Load the model parameter provided by the Timm library
state_dict = model_zoo.load_url('http://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=False, map_location=device)

# Image transformation for ImageNet ILSVRC 2012.
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# For convert the input image into original distribution to display
unnormalize = transforms.Compose([
    transforms.Normalize([0., 0., 0.], [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.,])
])


# Open the input image and transform
image = PIL.Image.open(IMAGE_PATH)
image = transform(image)
image = image.unsqueeze(0).to(device)

class_num=1000

print("[DEBUG] - MODEL: ", MODEL)
print("[DEBUG] - device: ", device)
print("[DEBUG] - image.shape: ", image.shape)
print("[DEBUG] - class_num: ", class_num)

def print_heatmap(heatmaps):
    fig, axs = plt.subplots(12,12, figsize=(70, 70))

    for i in range(12):
        for j in range(12):
            heatmap = heatmaps[0][i, j]
            heatmap = heatmap.reshape(1, 1, 14, 14)
            heatmap = transforms.Resize((224, 224))(heatmap[0])
            heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min())
            heatmap = heatmap.detach().cpu().numpy()
            heatmap = np.transpose(heatmap, (1, 2, 0))

            axs[i, j].imshow(heatmap)
            axs[i, j].axis('off')

# Models and Methods
model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
model.load_state_dict(state_dict, strict=True)
model.eval()

agc_method = AGCAM(model)
betterAGC_method = Better_AGCAM(model)

# Generate heatmap using AGCAM method
with torch.enable_grad():
    prediction, agc_heatmap = agc_method.generate(image)
    agc_heatmap = transforms.Resize((224, 224))(agc_heatmap[0])
    agc_heatmap = (agc_heatmap - agc_heatmap.min())/(agc_heatmap.max()-agc_heatmap.min())
    agc_heatmap = agc_heatmap.detach().cpu().numpy()
    agc_heatmap = np.transpose(agc_heatmap, (1, 2, 0))

# Generate heatmap using BetterAGCAM method
with torch.enable_grad():
    prediction, better_agc_heatmap, output_truth = betterAGC_method.generate(image)

    tensor_heatmaps = better_agc_heatmap[0]
    tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
    tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
    
    # Compute min and max along each image
    min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
    max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
    # Normalize using min-max scaling
    tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
  
    m = torch.mul(tensor_heatmaps, image)
    
    with torch.no_grad():
        output_mask = model(m)

    heatmaps = better_agc_heatmap[0]
  
    agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
    agc_scores = torch.sigmoid(agc_scores)
    agc_scores = agc_scores.reshape(heatmaps.shape[0], heatmaps.shape[1])
    
    my_cam = (agc_scores.view(12, 12, 1, 1, 1) * heatmaps).sum(axis=(0, 1))
    mask = my_cam
    mask = mask.unsqueeze(0)
    upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
    mask = upsample(mask)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    sigmoid_heatmap = mask.detach().cpu().numpy()[0]
    sigmoid_heatmap = np.transpose(sigmoid_heatmap, (1, 2, 0))


# Display final heatmap
image = unnormalize(image)
image = image.detach().cpu().numpy()[0]
image = np.transpose(image, (1, 2, 0))
fig, axs = plt.subplots(1,3, figsize=(15, 70))
axs[0].set_title(classes[prediction.item()])
axs[0].imshow(image)
axs[0].axis('off')

axs[1].set_title('AGCAM')
axs[1].imshow(image)
axs[1].imshow(agc_heatmap, cmap='jet', alpha=0.5)
axs[1].axis('off')

axs[2].set_title('Better AGCAM using sigmoid')
axs[2].imshow(image)
axs[2].imshow(sigmoid_heatmap, cmap='jet', alpha=0.5)
axs[2].axis('off')
