import pickle
import torch
from  ViTDet.models.detection.mask_rcnn import maskrcnn_vit_b_16_sfpn

path = './model_final_61ccd1.pkl'

with open(path, 'rb') as f:
    weights = pickle.load(f)

# print(weights['model'].keys())



model = maskrcnn_vit_b_16_sfpn().to('cuda')
model.eval()
x = [torch.rand(3, 300, 400).to('cuda'), torch.rand(3, 500, 400).to('cuda')]
predictions = model(x)

print(predictions)