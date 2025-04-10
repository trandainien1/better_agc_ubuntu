# import os

# def read_bounding_boxes(annotation_file="bounding_box.txt"):
#     """Reads bounding box data from a single text file and returns a dictionary."""
#     if not os.path.exists(annotation_file):
#         return {}  # Return empty dictionary if the file doesn't exist
    
#     bounding_boxes = {}
#     with open(annotation_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 5:
#                 image_id, x, y, w, h = parts
#                 bounding_boxes[image_id] = list(map(float, (x, y, w, h)))
    
#     return bounding_boxes

# a = read_bounding_boxes(r"C:\Users\Asus\Downloads\CUB_200_2011\CUB_200_2011\bounding_boxes.txt")
# print(a['2'])

import torch

a = torch.tensor([])

b = torch.tensor([1,2,3,4]).unsqueeze(0)
c = torch.tensor([1,2,3,4]).unsqueeze(0)

a = torch.cat((a, b))
a = torch.cat((a, c))

print(a)