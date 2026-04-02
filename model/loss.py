import torch
import torch.nn.functional as F


def compute_loss(node_weights, gt_mask):
    """
    Loss function to encourage target node weight close to 1 and non-target node weights close to 0.
    Args:
        node_weights: (batch_size, num_nodes) model output node weights
        gt_mask: (batch_size, num_nodes) binary mask indicating target nodes
    Returns:
        loss: scalar loss value
    """
    mse_loss = 0.0
    gt_mask_max = torch.max(gt_mask, dim=-1, keepdim=True)[0].expand_as(gt_mask)
    gt_mask_hard = (gt_mask > gt_mask_max * 0.5)
    for i in range(gt_mask.shape[0]):
        if torch.sum(gt_mask_hard[i]) > 0:
            mse_loss += F.mse_loss(node_weights[i, gt_mask_hard[i]], gt_mask[i, gt_mask_hard[i]]) / torch.sum(gt_mask_hard[i]).float()
        if torch.sum(~gt_mask_hard[i]) > 0:
            mse_loss += F.mse_loss(node_weights[i, ~gt_mask_hard[i]], gt_mask[i, ~gt_mask_hard[i]]) / torch.sum(~gt_mask_hard[i]).float()
    return mse_loss
