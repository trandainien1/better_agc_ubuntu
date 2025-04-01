import os
from glob import glob
import PIL
import torch
from torchvision.datasets import ImageFolder
from bs4 import BeautifulSoup

class ImageNetDataset_val(ImageFolder):
    def __init__(self, root_dir, transforms=None):
        self.img_dir = os.path.join(root_dir, "Data", "CLS-LOC", "val")
        self.annotation_dir = os.path.join(root_dir, "Annotations", "CLS-LOC", "val")
        self.classes = sorted(os.listdir(self.img_dir))
        # self.classes = os.listdir(self.img_dir)
        self.transforms = transforms
        self.img_data = []
        self.img_labels = []

        for idx, cls in enumerate(self.classes):
            # self.class_name.append(cls)
            img_cls_dir = os.path.join(self.img_dir, cls)

            for img in glob(os.path.join(img_cls_dir, '*.JPEG')):
                self.img_data.append(img)
                self.img_labels.append(idx)


    def __getitem__(self, idx):
        img_path, label = self.img_data[idx], self.img_labels[idx]
        # print('[DEBUG]', img_path)
        img = PIL.Image.open(img_path).convert('RGB')
        # img.show()
        width, height = img.size
        img_name = img_path.split('/')[-1].split('.')[0]
        anno_path = os.path.join(self.annotation_dir, img_name+".xml")
        with open(anno_path, 'r') as f:
            file = f.read()
        soup = BeautifulSoup(file, 'html.parser')
        if self.transforms:
            img = self.transforms(img)
        objects = soup.findAll('object')
        
        bnd_box = torch.tensor([])

        for object in objects:
            xmin = int(object.bndbox.xmin.text)
            ymin = int(object.bndbox.ymin.text)
            xmax = int(object.bndbox.xmax.text)
            ymax = int(object.bndbox.ymax.text)
            xmin = int(xmin/width*224)
            ymin = int(ymin/height*224)
            xmax = int(xmax/width*224)
            ymax = int(ymax/height*224)
            if bnd_box.dim()==1:
                bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
            else:
                bnd_box = torch.cat((bnd_box, torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)), dim=0)
        # print(bnd_box.shape)
        sample = {
            'image': img, 
            'label': label, 
            'filename': img_name, 
            'num_objects': len(objects), 
            'bnd_box': bnd_box, 
            'img_path': img_path
            }
        return sample

    def __len__(self):
        return len(self.img_data)
    
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    annotation_folder = 'CUB_200_2011/bounding_boxes'  # Assuming annotation TXT files are here
    
    def __init__(self, root, train=True, transform=None, loader=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader if loader else self.default_loader
        self.train = train
        
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _load_bounding_boxes(self, img_id):
        annotation_path = os.path.join(self.root, self.annotation_folder, f'{img_id}.txt')
        
        if not os.path.exists(annotation_path):
            return torch.tensor([])  # Return empty tensor if no annotation exists
        
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        bnd_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, x, y, w, h = map(float, parts)
                xmin, ymin, xmax, ymax = x, y, x + w, y + h
                bnd_boxes.append([xmin, ymin, xmax, ymax])
                # bnd_boxes.append([x, y, w, h])
        
        return torch.tensor(bnd_boxes)
    
    def default_loader(self, path):
        return Image.open(path).convert('RGB')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Convert label from 1-based to 0-based index
        img = self.loader(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        img_name = sample.filepath.split('/')[-1].split('.')[0]
        bnd_box = self._load_bounding_boxes(sample.img_id)
        
        sample_dict = {
            'image': img,
            'label': target,
            'filename': img_name,
            'num_objects': len(bnd_box),
            'bnd_box': bnd_box,
            'img_path': img_path
        }
        return sample_dict