import os
import json
import torch
from tqdm import tqdm
import argparse


def gen_gnn_feat(dataset, split, pruning, base_path):
    max_obj = 100
    feat_dim = 512
    graph_file = os.path.join(base_path, 'preds', f'infer_{dataset}_{split}_m3d_{pruning}.pt')
    graph = torch.load(graph_file, weights_only=False)
    if dataset == 'scanrefer':
        vl_data_file = os.path.join('annotations', f'scanrefer_mask3d_{split}.json')
    elif dataset == 'scanqa':
        vl_data_file = os.path.join('annotations', f'scanqa_{split}.json')
    elif dataset == 'multi3drefer':
        vl_data_file = os.path.join('annotations', f'multi3dref_mask3d_{split}.json')
    elif dataset == 'scan2cap':
        vl_data_file = os.path.join('annotations', f'scan2cap_mask3d_{split}.json')
    vl_data = json.load(open(vl_data_file, 'r'))
    scan_split = split

    if scan_split == 'test':
        print("Skipping test scans...")
        return
    
    scans_list_file = os.path.join('data', 'scannet_metadata', f'scannetv2_{scan_split}.txt')
    scans_list = sorted([line.strip() for line in open(scans_list_file, 'r')])

    gnn_feat_comp = {}
    print('Loading GNN features...')
    for scan in tqdm(scans_list):
        scene_feat_file = os.path.join('annotations', 'output_vlsat', f'{scan}.pt')
        if not os.path.exists(scene_feat_file):
            print(f"[VLSAT] Scene graph for {scan} not found!")
            continue
        gnn_feat_raw = torch.load(scene_feat_file, weights_only=False)
        gnn_feat_comp[scan] = torch.zeros(max_obj, max_obj, feat_dim)
        for k, v in gnn_feat_raw.items():
            parts = k.split('_')
            gnn_feat_comp[scan][int(parts[1]), int(parts[4]), :] = v

    backup_file = os.path.join('annotations', 'scannet_gt_train_gnn_feats_2.pt')
    backup = torch.load(backup_file, weights_only=True)

    result = []
    print(f'Computing GNN features for 3D-VL dataset: {dataset}...')
    for idx, datum in tqdm(enumerate(vl_data), total=len(vl_data)):
        # datum['graph'] = graph[idx]
        scene_id = datum['scene_id']
        scene_graph = graph[idx]
        scene_graph_feat = []
        if scene_id not in gnn_feat_comp.keys():
            print(f"{scene_id} not found, using KNN scene feature...")
            _bkp = backup[scene_id]
            padding = torch.zeros((max_obj * scene_graph.shape[1] - _bkp.shape[0], feat_dim))
            result.append(torch.cat([_bkp, padding], dim=0))
        else:
            for u in range(min(scene_graph.shape[0], max_obj)):
                for v in scene_graph[u]:
                    scene_graph_feat.append(gnn_feat_comp[scene_id][u, v, :])
            scene_graph_feat = torch.stack(scene_graph_feat, dim=0)
            if scene_graph.shape[0] < max_obj:
                padding = torch.zeros((max_obj * scene_graph.shape[1] - scene_graph_feat.shape[0], feat_dim))
                scene_graph_feat = torch.cat([scene_graph_feat, padding], dim=0)
            result.append(scene_graph_feat)
    result = torch.stack(result, dim=0)
    os.makedirs(os.path.join(base_path, 'gnn_feat'), exist_ok=True)
    output_file = os.path.join(base_path, 'gnn_feat', f'{dataset}_m3d_{split}_gnn_feat_{pruning}.pt')
    torch.save(result, output_file)
    print(f'Saved GNN features to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scanqa')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--pruning', type=str, default='knn')
    parser.add_argument('--base_path', type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    pruning = args.pruning
    base_path = args.base_path

    gen_gnn_feat(dataset, split, pruning, base_path)