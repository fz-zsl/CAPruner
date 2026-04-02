import torch
import torch.nn as nn
import torch.nn.init as init

from model.node_distance import NodeDistance


class GraphTargetDetector(nn.Module):
    def __init__(self, hidden_dim=16, p=2.0, dist_metric: NodeDistance = None):
        """
        Args:
            hidden_dim: number of hidden units in edge weight network
            p: the parameter for p-norm aggregation
        """
        super().__init__()
        self.p = p
        self.dist_metric = dist_metric
        
        # 1. Network to compute edge weights w_ij from (a_i, a_j, d_ij)
        self.edge_weight_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 2. Node weight aggregation parameters (trainable)
        self.node_agg_weight = nn.Parameter(torch.ones(1))  # scaling parameter for aggregation
        self.node_bias = nn.Parameter(torch.zeros(1))       # bias parameter for aggregation

        # 3. Random initialization
        for m in self.edge_weight_net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
                if m.bias is not None:
                    init.uniform_(m.bias, a=-0.2, b=0.2)

    def calc_dist(self, bboxes):
        # coords: coordinates, (batch_size, num_nodes, 3)
        dist_matrix = self.dist_metric.calc_dist(bboxes)  # (batch_size, num_nodes, num_nodes)
        dist_norm = dist_matrix / (dist_matrix.max(dim=-1, keepdim=True)[0] + 1e-8)
        dist = torch.sigmoid(dist_norm)  # (batch_size, num_nodes, num_nodes)
        dist_valid = torch.nan_to_num(dist, nan=1.0)
        return dist_valid

    def forward(self, node_attr, bbox, n_obj):
        """
        Forward pass: compute final weights for each node
        Args:
            node_attr: (batch_size, num_nodes) node attributes
            bbox: (batch_size, num_nodes, 6) node bounding boxes
            n_obj: (batch_size,) number of objects per sample
        Returns:
            node_weights: (batch_size, num_nodes) final weights for each node
            edge_weights: (batch_size, num_nodes, num_nodes) all edge weights
        """
        eps = 1e-8
        batch_size, num_nodes = node_attr.shape[:2]

        # 1. Prepare node attribute pairs for edge weight computation
        attr1 = node_attr.unsqueeze(-1).repeat(1, 1, num_nodes)  # (batch_size, num_nodes, num_nodes)
        attr2 = node_attr.unsqueeze(1).repeat(1, num_nodes, 1)  # (batch_size, num_nodes, num_nodes)
        # 2. Compute pairwise distances
        dist = self.calc_dist(bbox).detach()  # (batch_size, num_nodes, num_nodes)

        # 3. Compute edge weights
        edge_input = torch.stack([attr1, attr2, dist], dim=-1)  # (batch_size, num_nodes, num_nodes, 3)
        edge_input_flat = edge_input.reshape(batch_size * num_nodes * num_nodes, 3)  # (batch_size*num_nodes*num_nodes, 3)

        edge_weights_flat = self.edge_weight_net(edge_input_flat)  # (batch_size*num_nodes*num_nodes, 1)
        edge_weights = edge_weights_flat.reshape(batch_size, num_nodes, num_nodes) + eps  # (batch_size, num_nodes, num_nodes)
        edge_weights_clamp = torch.clamp(edge_weights, min=eps, max=1.0)  # avoid zero weights
        
        # 4. Aggregate edge weights to get node weights (p-norm + trainable parameters)
        # For each node u, aggregate all connected edge weights: sum(w_iu^p)^(1/p) (p-norm, balancing max and sum)
        edge_weights_p = edge_weights_clamp.pow(self.p)  # (batch_size, num_nodes, num_nodes)
        node_agg = edge_weights_p.sum(dim=-1)  # Sum over all j: w_u = sum(w_uj^p), (batch_size, num_nodes)
        node_agg = node_agg.pow(1/self.p)  # Take p-th root to restore scale

        # 5. Apply trainable parameters to adjust node weights
        node_weights = self.node_agg_weight * node_agg + self.node_bias
        node_weights_sigmoid = torch.sigmoid(node_weights)  # Normalize to [0,1]

        # 6. Masking based on number of objects, keep [0, n_obj) nodes, set others to 0

        # For node_weights_sigmoid masking
        node_mask = torch.zeros_like(node_weights_sigmoid, dtype=torch.bool)
        for i in range(batch_size):
            node_mask[i, n_obj[i]:] = True  # Mark positions to be zeroed
        node_weights_sigmoid = node_weights_sigmoid.masked_fill(node_mask, 0.0)

        # For edge_weights masking (assuming edge_weights is [batch_size, num_nodes, num_nodes])
        edge_mask = torch.zeros_like(edge_weights, dtype=torch.bool)
        for i in range(batch_size):
            edge_mask[i, n_obj[i]:, :] = True
            edge_mask[i, :, n_obj[i]:] = True
        edge_weights = edge_weights.masked_fill(edge_mask, 0.0)
        
        return node_weights_sigmoid, edge_weights