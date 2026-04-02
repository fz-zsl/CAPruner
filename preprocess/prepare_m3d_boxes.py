import torch
from tqdm import tqdm
import json
import os

files = [
    "annotations/scannet_mask3d_train_attributes.pt",
    "annotations/scannet_mask3d_val_attributes.pt",
    "annotations/scannet_mask3d_test_attributes.pt",
]

def construct_bbox(bbox):
    x, y, z, sx, sy, sz = bbox.tolist()
    sx /= 2
    sy /= 2
    sz /= 2
    return [x - sx, y - sy, z - sz, x + sx, y + sy, z + sz]

os.makedirs("data/m3d", exist_ok=True)
for file in files:
    attr = torch.load(file, weights_only=True)
    for scan_id in tqdm(attr.keys()):
        result = []
        for idx, (label, bbox) in enumerate(zip(attr[scan_id]["objects"], attr[scan_id]["locs"])):
            new_bbox = construct_bbox(bbox)
            result.append({'id': idx, 'label': label, 'bbox': new_bbox})
        with open(os.path.join('data', 'm3d', f'{scan_id}_m3d_boxes.json'), 'w') as f:
            json.dump(result, f, indent=4)
