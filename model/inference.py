import argparse
import os
import torch
import yaml
from typing import Dict, Any
from tqdm import tqdm
import logging
import sys
from datetime import datetime

from model.node_distance import LpDistance
from model.target_detector import GraphTargetDetector
from model.dataset import GraphDataset
from model.pruning import KNNPruning, MSTPruning


def setup_logger(log_file, level=logging.INFO):
    logger = logging.getLogger("ScenePuruneLogger")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Default values for model config
    model_defaults = {'node_val': 'soft', 'hidden_dim': 16, 'agg_p': 2.0, 'dist_metric': 'lp', 'dist_p': 2.0}
    for key, default in model_defaults.items():
        if key not in config.get('model', {}):
            config['model'][key] = default
    
    # Default values for training config
    training_defaults = {'batch_size': 16, 'max_obj': 150}
    for key, default in training_defaults.items():
        if key not in config.get('training', {}):
            config['training'][key] = default
    
    # Default values for inference config
    inference_defaults = {'split': 'val', 'pruning': 'knn', 'conn': 2}
    for key, default in inference_defaults.items():
        if key not in config.get('inference', {}):
            config['inference'][key] = default

    logger.info("Model config:", config['model'])
    logger.info("Training config:", config['training'])
    logger.info("Inference config:", config['inference'])

    return config


def inference(config: Dict[str, Any], logger: logging.Logger, dataset_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Inference on device: {device}")

    # Initialize model with config parameters
    if config['model']['dist_metric'] == 'lp':
        dist_metric = LpDistance(config['model']['dist_p'])
    else:
        raise ValueError(f"Unknown distance metric: {config['model']['dist_metric']}")
    model = GraphTargetDetector(
        hidden_dim=config['model']['hidden_dim'],
        p=config['model']['agg_p'],
        dist_metric=dist_metric,
    ).to(device)

    if 'pretrained' in config['model'] and config['model']['pretrained'] is not None:
        model_pth = config['model']['pretrained']
        if os.path.exists(model_pth):
            model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
            logger.info(f"Loaded pretrained model from {model_pth}")
        else:
            assert False, f"Pretrained model path {model_pth} does not exist."
    else:
        assert False, "No pretrained model specified."

    # Prepare dataset and dataloader
    dataset = GraphDataset(config['inference']['split'], config, dataset_name=dataset_name, use_gt=False)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=32
    )

    with torch.no_grad():
        pruned_edges = []
        max_obj = config['training']['max_obj']
        conn = config['inference']['conn']
        model.eval()
        logger.info(f'Inferencing on {config["inference"]["split"]} set...')
        for node_val, bbox, n_obj, gt_mask, fg_mask in tqdm(dataloader):
            node_val, bbox = node_val.to(device), bbox.to(device)
            n_obj, gt_mask, fg_mask = n_obj.to(device), gt_mask.to(device), fg_mask.to(device)
            
            _, edge_weights = model(node_val, bbox, n_obj)
            symm_edge_weights = (edge_weights + edge_weights.transpose(1, 2)) / 2
            symm_edge_weights[:, torch.arange(max_obj), torch.arange(max_obj)] = -float('inf')
            symm_edge_weights = torch.nan_to_num(symm_edge_weights, nan=-float('inf'))

            if config['inference']['pruning'] == 'knn':
                pruning_method = KNNPruning()
            elif config['inference']['pruning'] == 'mst':
                pruning_method = MSTPruning()
            else:
                raise ValueError(f"Unknown pruning method: {config['inference']['pruning']}")
            pruned_edges.extend(pruning_method.prune(symm_edge_weights, n_obj, max_obj, conn, fg_mask))

        pruned_edges = torch.stack(pruned_edges, dim=0)
        split = config['inference']['split']
        seg = config['dataset']['seg_type']
        pruning = config['inference']['pruning']
        torch.save(pruned_edges, os.path.join(config['inference']['save_dir'], f"infer_{dataset_name}_{split}_{seg}_{pruning}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration YAML file')
    args = parser.parse_args()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('logs', f'infer_{time_str}_{args.config.split("/")[-1].split(".")[0]}.log')
    logger = setup_logger(log_file)
    config = load_config(args.config, logger)
    save_dir = os.path.join('outputs', config['model']['pretrained'].split('/')[1], 'preds')
    os.makedirs(save_dir, exist_ok=True)
    config['inference']['save_dir'] = save_dir
    for dataset_name in config['inference']['infer_tag'].split('_'):
        inference(config, logger, dataset_name)
