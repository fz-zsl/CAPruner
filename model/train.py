import argparse
import os
import torch
import torch.optim as optim
import yaml
from typing import Dict, Any
from tqdm import tqdm
from datetime import datetime
import logging
import sys
import json

from model.node_distance import LpDistance
from model.target_detector import GraphTargetDetector
from model.loss import compute_loss
from model.dataset import GraphDataset


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
    training_defaults = {'batch_size': 16, 'max_obj': 150, 'lr': 1e-3, 'epochs': 10, 'val_freq': 3}
    for key, default in training_defaults.items():
        if key not in config.get('training', {}):
            config['training'][key] = default

    logger.info("Model config:", config['model'])
    logger.info("Training config:", config['training'])

    return config


def train(config: Dict[str, Any], logger: logging.Logger, val_only: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    if not val_only:
        with open(os.path.join(config['train_path'], f'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

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

    if 'pretrained' in config['model']:
        model_pth = config['model']['pretrained']
        if os.path.exists(model_pth):
            model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
            logger.info(f"Loaded pretrained model from {model_pth}")
        else:
            logger.info(f"Pretrained model path {model_pth} does not exist. Training from scratch.")
            assert not val_only, "Pretrained model not found for validation."
    else:
        logger.info("No pretrained model specified. Training from scratch.")
        assert not val_only, "Pretrained model not found for validation."

    # Prepare dataset and dataloader
    if not val_only:
        train_datasets = []
        for dataset_name in config['training']['train_tag'].split('_'):
            train_datasets.append(GraphDataset('train', config, dataset_name))
        train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(train_datasets),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=32
        )
    val_datasets = []
    for dataset_name in config['training']['val_tag'].split('_'):
        val_datasets.append(GraphDataset('val', config, dataset_name))
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_datasets),
        batch_size=config['training']['batch_size'],
        shuffle=False,  # previously was True
        num_workers=32
    )

    if val_only:
        logger.info(f"Evaluating on validation set...")
        val(val_dataloader, model, device, logger)
        return

    # Initialize optimizer and scheduler
    total_steps = config['training']['epochs'] * len(train_dataloader)
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # Training loop
    model.train()
    torch.autograd.set_detect_anomaly(True)
    best_val_acc = 0.0
    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, lr={scheduler.get_last_lr()[0]}")
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        for node_val, pos, n_obj, gt_mask, fg_mask in tqdm(train_dataloader):  # tqdm()
            node_val, pos = node_val.to(device), pos.to(device)
            n_obj, gt_mask, fg_mask = n_obj.to(device), gt_mask.to(device), fg_mask.to(device)
            
            optimizer.zero_grad()
            node_weights, _ = model(node_val, pos, n_obj)
            total_loss = compute_loss(node_weights, gt_mask.float())
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            gt_target = gt_mask & fg_mask
            target_cnt = gt_target.sum(dim=1)
            node_weights_masked = torch.where(fg_mask, node_weights, -torch.inf)
            _, indices = torch.topk(node_weights_masked, k=node_weights.shape[1], dim=1)
            pred_target = torch.zeros_like(node_weights, dtype=torch.bool)
            for i in range(node_weights.shape[0]):
                pred_target[i, indices[i, :target_cnt[i]]] = True
            epoch_acc += (gt_target == pred_target).all(dim=1).float().mean().item()
        
        avg_loss = epoch_loss / len(train_dataloader)
        avg_acc = epoch_acc / len(train_dataloader) * 100.0
        logger.info(f"[Epoch {epoch+1:3d}] Avg. Loss: {avg_loss:.4f};  Avg. Train Acc: {avg_acc:.4f}")

        if (epoch + 1) % config['training']['val_freq'] == 0:
            _, val_acc = val(val_dataloader, model, device, logger)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(config['train_path'], f'best_model.pth'))
                logger.info(f"[!!] New best model saved with Val Acc: {best_val_acc:.4f}")
    
        torch.save(model.state_dict(), os.path.join(config['train_path'], f'last_model.pth'))
    print(f"Training finished, best val Acc: {best_val_acc:.4f}")


def val(val_dataloader, model, device, logger):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for node_val, pos, n_obj, gt_mask, fg_mask in tqdm(val_dataloader):  # tqdm()
            node_val, pos = node_val.to(device), pos.to(device)
            n_obj, gt_mask, fg_mask = n_obj.to(device), gt_mask.to(device), fg_mask.to(device)
            
            node_weights, _ = model(node_val, pos, n_obj)
            total_loss = compute_loss(node_weights, gt_mask.float())
            
            val_loss += total_loss.item()
            gt_target = gt_mask & fg_mask
            target_cnt = gt_target.sum(dim=1)
            node_weights_masked = torch.where(fg_mask, node_weights, -torch.inf)
            _, indices = torch.topk(node_weights_masked, k=node_weights.shape[1], dim=1)
            pred_target = torch.zeros_like(node_weights, dtype=torch.bool)
            for i in range(node_weights.shape[0]):
                pred_target[i, indices[i, :target_cnt[i]]] = True
            val_acc += (gt_target == pred_target).all(dim=1).float().mean().item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_acc = val_acc / len(val_dataloader) * 100.0
    logger.info(f"Validation Avg. Loss: {avg_val_loss:.4f}; Avg. Val Acc: {avg_val_acc:.4f}")
    model.train()
    return avg_val_loss, avg_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration YAML file')
    parser.add_argument('--val', action='store_true', help='If set, run validation instead of training')
    args = parser.parse_args()
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_tag = args.config.split("/")[-1].split(".")[0]
    train_path = os.path.join('outputs', f'{time_str}_{train_tag}') if not args.val else None
    if args.val:
        os.makedirs('logs', exist_ok=False)
        log_file = os.path.join('logs', f'val_{time_str}_{train_tag}.log')
    else:
        os.makedirs(train_path, exist_ok=False)
        log_file = os.path.join(train_path, f'train_{time_str}_{train_tag}.log')
    logger = setup_logger(log_file)
    config = load_config(args.config, logger)
    config['time_str'] = time_str
    config['train_path'] = train_path
    train(config, logger, val_only=args.val)
