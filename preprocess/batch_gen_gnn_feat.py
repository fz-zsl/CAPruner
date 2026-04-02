from gen_gnn_feat import gen_gnn_feat
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    args = parser.parse_args()
    base_path = args.base_path
    pred_path = os.path.join(base_path, 'preds')
    for filename in os.listdir(pred_path):
        if not filename.endswith('.pt'):
            continue
        _filename = filename.split('.')[0]
        dataset = _filename.split('_')[1]
        split = _filename.split('_')[2]
        pruning = _filename.split('_')[-1]
        target_file_name = f'{dataset}_m3d_{split}_gnn_feat_{pruning}.pt'
        if os.path.exists(os.path.join(base_path, 'gnn_feat', target_file_name)):
            continue
        gen_gnn_feat(dataset, split, pruning, base_path)
