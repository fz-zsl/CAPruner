import os
import json
import torch
from typing import Dict, Any
from tqdm import tqdm
import re

from model.node_matching import HardCategNodeVal, SoftCategNodeVal, NoNodeVal


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])

    interVolume = max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)
    boxAVolume = (boxA[3] - boxA[0]) * (boxA[4] - boxA[1]) * (boxA[5] - boxA[2])
    boxBVolume = (boxB[3] - boxB[0]) * (boxB[4] - boxB[1]) * (boxB[5] - boxB[2])
    iou = interVolume / float(boxAVolume + boxBVolume - interVolume + 1e-8)
    return iou


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, config: Dict[str, Any], dataset_name: str = 'scanrefer', use_gt: bool = True):
        max_obj = config['training']['max_obj']
        if config['model']['node_val'] == 'soft':
            self.node_val = SoftCategNodeVal()
        elif config['model']['node_val'] == 'hard':
            self.node_val = HardCategNodeVal()
        else:
            self.node_val = NoNodeVal()

        assert split in ['train', 'val'], "split must be 'train', or 'val'"
        print(f"Loading segmentations in the {split} dataset...")
        
        scan_split = split
        scan_file_path = config['dataset']['scannet'][f'{scan_split}_scans']
        with open(scan_file_path, 'r') as f:
            self.scan_list = [line.strip() for line in f.readlines()]
        self.data_dir = os.path.join(config['dataset']['data_root'], config['dataset']['seg_type'])

        print(f"Loading bounding boxes of {config['dataset']['seg_type']}...")
        self.scan = {}

        os.makedirs('capruner_cache', exist_ok=True)
        if os.path.exists(os.path.join('capruner_cache', f'foreground_{split}.pt')):
            foreground_ids = torch.load(os.path.join('capruner_cache', f'foreground_{split}.pt'), weights_only=True)
        else:
            foreground_ids = {}

        foreground_updated = False
        for scan in tqdm(self.scan_list):
            self.scan[scan] = {'categs': [], 'bboxes': []}
            bboxes_path = os.path.join(self.data_dir, f"{scan}_{config['dataset']['seg_type']}_boxes.json")
            if config['dataset']['seg_type'] == 'm3d' and not os.path.exists(bboxes_path):
                # print("Using GT boxes as M3D boxes are not found...")
                bboxes_path = os.path.join(config['dataset']['data_root'], 'gt', f'{scan}_gt_boxes.json')
            if not os.path.exists(bboxes_path):
                print(f"Warning: {config['dataset']['seg_type'].upper()} boxes for scan {scan} not found in {self.data_dir}")
                continue
            with open(bboxes_path, 'r') as f:
                self.scan[scan]['raw_bboxes'] = json.load(f)[:max_obj]
            self.scan[scan]['n_objs'] = min(len(self.scan[scan]['raw_bboxes']), max_obj)
            for bbox in self.scan[scan]['raw_bboxes']:
                self.scan[scan]['categs'].append(bbox['label'])
                self.scan[scan]['bboxes'].append(torch.tensor(bbox['bbox']))  # modified
            self.scan[scan]['bboxes'] = torch.stack(self.scan[scan]['bboxes'], dim=0)  # (num_nodes, 6)

            if scan in foreground_ids:
                self.scan[scan]['foreground_mask'] = foreground_ids[scan]
            else:
                self.scan[scan]['foreground_mask'] = torch.zeros(max_obj, dtype=torch.bool)
                for i in range(self.scan[scan]['n_objs']):
                    _keep_box = True
                    for j in range(i + 1, self.scan[scan]['n_objs']):
                        if compute_iou(self.scan[scan]['bboxes'][i], self.scan[scan]['bboxes'][j]) > 0.99:
                            _keep_box = False
                            break
                    if _keep_box:
                        self.scan[scan]['foreground_mask'][i] = True
                foreground_ids[scan] = self.scan[scan]['foreground_mask']
                foreground_updated = True
            
        if foreground_updated:
            torch.save(foreground_ids, os.path.join('capruner_cache', f'foreground_{split}.pt'))

        if config['dataset']['seg_type'] != 'gt' and use_gt:
            print(f"Loading ground truth bounding boxes...")
            self.gt_data_dir = os.path.join(config['dataset']['data_root'], 'gt')
            for scan in tqdm(self.scan_list):
                bboxes_path = os.path.join(self.gt_data_dir, f'{scan}_gt_boxes.json')
                if not os.path.exists(bboxes_path):
                    print(f"Warning: GT boxes for scan {scan} not found in {self.gt_data_dir}")
                    continue
                with open(bboxes_path, 'r') as f:
                    if scan not in self.scan:
                        print(f"Warning: Scan {scan} not found in loaded scans.")
                        continue
                    self.scan[scan]['gt_bboxes'] = json.load(f)[:max_obj]

        if dataset_name == 'scanrefer':
            # load utterances and gt from ScanRefer (adds dataset path in config)
            print(f"Loading the {split} split of ScanRefer...")
            with open(config['dataset']['scanrefer'][f'{split}_data'], 'r') as f:
                scanrefer = json.load(f)
            self.data = []
            for datum in tqdm(scanrefer):  # tqdm()
                scene_id = datum['scene_id']
                utterance = datum['description'] if 'description' in datum else datum['prompt'].split('\"')[1]
                node_attr = torch.zeros(max_obj)  # (max_obj, ) initialize node attribute tensor
                for idx, categ in enumerate(self.scan[scene_id]['categs']):
                    node_attr[idx] = self.node_val.categ_dist(categ, utterance)
                gt_mask = torch.zeros(max_obj, dtype=torch.bool)  # (max_obj, ) initialize gt mask
                if use_gt and not os.path.exists(os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt')):
                    caption_id = re.findall(r'<OBJ(\d+)>', datum.get('caption', datum.get('ref_captions', [''])[0]))[0]
                    gt_seg_ans = int(datum.get('object_id', caption_id))
                    gt_mask[gt_seg_ans] = True
                pos = torch.zeros((max_obj, 6))  # (max_obj, 6) initialize bounding boxes
                pos[:self.scan[scene_id]['n_objs'], :] = self.scan[scene_id]['bboxes']  # fill in valid positions
                new_datum = {
                    'scene_id': scene_id,
                    'node_attr': node_attr,  # (max_obj,) node attribute values
                    'bboxes': pos,  # (max_obj, 6) node positions
                    'n_obj': torch.tensor(self.scan[scene_id]['n_objs']),  # (1,) number of objects
                    'gt_mask': gt_mask,  # (max_obj,) binary mask indicating target nodes, try here
                    'fg_mask': self.scan[scene_id]['foreground_mask']
                }
                self.data.append(new_datum)
        elif dataset_name == 'scanqa':
            # load utterances and gt from ScanQA (adds dataset path in config)
            print(f"Loading the {split} split of ScanQA...")
            with open(config['dataset']['scanqa'][f'{split}_data'], 'r') as f:
                scanqa = json.load(f)
            self.data = []
            for datum in tqdm(scanqa):  # tqdm()
                scene_id = datum['scene_id']
                utterance = datum['question'] if 'question' in datum else datum['prompt'].split(' Answer the question using')[0]
                node_attr = torch.zeros(max_obj)  # (max_obj, ) initialize node attribute tensor
                for idx, categ in enumerate(self.scan[scene_id]['categs']):
                    node_attr[idx] = self.node_val.categ_dist(categ, utterance)
                gt_mask = torch.zeros(max_obj, dtype=torch.bool)  # (max_obj, ) initialize gt mask
                if use_gt and not os.path.exists(os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt')):
                    gt_seg_ans = [int(item) for item in datum['object_ids']] if 'object_ids' in datum else [int(datum['obj_id'])]
                    for _obj in gt_seg_ans:
                        _new_obj = -1
                        _max_iou = 0.0
                        _gt_box = self.scan[scene_id]['gt_bboxes'][_obj]['bbox']
                        _bboxes = self.scan[scene_id]['bboxes']
                        for i in range(len(_bboxes)):
                            _new_iou = compute_iou(_gt_box, _bboxes[i])
                            if _new_iou > _max_iou:
                                _max_iou = _new_iou
                                _new_obj = i
                        gt_mask[_new_obj] = True
                pos = torch.zeros((max_obj, 6))  # (max_obj, 6) initialize bounding boxes
                pos[:self.scan[scene_id]['n_objs'], :] = self.scan[scene_id]['bboxes']  # fill in valid positions
                new_datum = {
                    'scene_id': scene_id,
                    'node_attr': node_attr,  # (max_obj,) node attribute values
                    'bboxes': pos,  # (max_obj, 6) node positions
                    'n_obj': torch.tensor(self.scan[scene_id]['n_objs']),  # (1,) number of objects
                    'gt_mask': gt_mask,  # (max_obj,) binary mask indicating target nodes, try here
                    'fg_mask': self.scan[scene_id]['foreground_mask']
                }
                self.data.append(new_datum)
        elif dataset_name == 'multi3drefer':
            # load utterances and gt from Multi3DRefer (adds dataset path in config)
            print(f"Loading the {split} split of Multi3DRefer...")
            with open(config['dataset']['multi3drefer'][f'{split}_data'], 'r') as f:
                multi3dref = json.load(f)
            self.data = []
            for datum in tqdm(multi3dref):  # tqdm()
                scene_id = datum['scene_id']
                utterance = datum['description'] if 'description' in datum else datum['prompt'].split('\"')[1]
                node_attr = torch.zeros(max_obj)  # (max_obj, ) initialize node attribute tensor
                for idx, categ in enumerate(self.scan[scene_id]['categs']):
                    node_attr[idx] = self.node_val.categ_dist(categ, utterance)
                gt_mask = torch.zeros(max_obj, dtype=torch.bool)  # (max_obj, ) initialize gt mask
                if use_gt and not os.path.exists(os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt')):
                    if 'object_ids' in datum:
                        gt_seg_ans = [int(item) for item in datum['object_ids']]
                        for _obj in gt_seg_ans:
                            gt_mask[_obj] = True
                    elif 'ref_captions' in datum:
                        gt_seg_ans = [int(item) for item in datum['ref_captions']]
                        for _obj in gt_seg_ans:
                            _new_obj = -1
                            _max_iou = 0.0
                            _gt_box = self.scan[scene_id]['gt_bboxes'][_obj]['bbox']
                            _bboxes = self.scan[scene_id]['bboxes']
                            for i in range(len(_bboxes)):
                                _new_iou = compute_iou(_gt_box, _bboxes[i])
                                if _new_iou > _max_iou:
                                    _max_iou = _new_iou
                                    _new_obj = i
                            if _new_obj >= 0:
                                gt_mask[_new_obj] = True
                    else:
                        num_str_list = re.findall(r'<OBJ(\d+)>', datum['caption'])
                        gt_seg_ans = [int(num_str) for num_str in num_str_list]
                        for _obj in gt_seg_ans:
                            gt_mask[_obj] = True
                pos = torch.zeros((max_obj, 6))  # (max_obj, 6) initialize bounding boxes
                pos[:self.scan[scene_id]['n_objs'], :] = self.scan[scene_id]['bboxes']  # fill in valid positions
                new_datum = {
                    'scene_id': scene_id,
                    'node_attr': node_attr,  # (max_obj,) node attribute values
                    'bboxes': pos,  # (max_obj, 6) node positions
                    'n_obj': torch.tensor(self.scan[scene_id]['n_objs']),  # (1,) number of objects
                    'gt_mask': gt_mask,  # (max_obj,) binary mask indicating target nodes, try here
                    'fg_mask': self.scan[scene_id]['foreground_mask']
                }
                self.data.append(new_datum)
        elif dataset_name == 'scan2cap':
            # load utterances and gt from Scan2Cap (adds dataset path in config)
            print(f"Loading the {split} split of Scan2Cap...")
            with open(config['dataset']['scan2cap'][f'{split}_data'], 'r') as f:
                scan2cap = json.load(f)
            self.data = []
            for datum in tqdm(scan2cap):  # tqdm()
                scene_id = datum['scene_id']
                node_attr = torch.zeros(max_obj)  # (max_obj, ) initialize node attribute tensor
                node_attr[int(datum['pred_id'])] = 1.0
                gt_mask = torch.zeros(max_obj, dtype=torch.bool)  # (max_obj, ) initialize gt mask
                gt_mask[int(datum['pred_id'])] = True
                pos = torch.zeros((max_obj, 6))  # (max_obj, 6) initialize bounding boxes
                pos[:self.scan[scene_id]['n_objs'], :] = self.scan[scene_id]['bboxes']  # fill in valid positions
                new_datum = {
                    'scene_id': scene_id,
                    'node_attr': node_attr,  # (max_obj,) node attribute values
                    'bboxes': pos,  # (max_obj, 6) node positions
                    'n_obj': torch.tensor(self.scan[scene_id]['n_objs']),  # (1,) number of objects
                    'gt_mask': gt_mask,  # (max_obj,) binary mask indicating target nodes, try here
                    'fg_mask': self.scan[scene_id]['foreground_mask']
                }
                self.data.append(new_datum)
        else:
            raise NotImplementedError(f"Unrecognizable dataset: {dataset_name}")

        if os.path.exists(os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt')):
            gt_masks = torch.load(os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt'), weights_only=True)
            for i in range(len(self.data)):
                self.data[i]['gt_mask'] = gt_masks[i]
        elif dataset_name != 'scan2cap':
            gt_masks = []
            for datum in tqdm(self.data):
                for i in range(max_obj):
                    if not datum['fg_mask'][i]:
                        continue
                    for j in range(max_obj):
                        if datum['gt_mask'][j] and compute_iou(datum['bboxes'][i], datum['bboxes'][j]) > 0.99:
                            datum['gt_mask'][i] = True
                datum['gt_mask'] = datum['fg_mask'] & datum['gt_mask']
                gt_masks.append(datum['gt_mask'])
            gt_masks = torch.stack(gt_masks, dim=0)
            torch.save(gt_masks, os.path.join('capruner_cache', f'gt_{config["dataset"]["seg_type"]}_{dataset_name}_{split}.pt'))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        return datum['node_attr'], datum['bboxes'], datum['n_obj'], datum['gt_mask'], datum['fg_mask']


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='val', max_obj=150, use_gt=True):
        super().__init__()
        self.config = config
        self.split = split
        self.max_obj = max_obj
        self.use_gt = use_gt
        self.node_val = NodeVal(config['model']['node_val'])
        self.dataset = GraphDataset(config, split, max_obj, use_gt)