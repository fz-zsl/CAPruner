import torch
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod


class PruningMethod(ABC):
    @abstractmethod
    def prune(self, edge_weights, n_obj, max_obj, conn, foreground_mask) :
        """
        Prune edges based on the given edge weights and number of objects.

        Parameters:
        edge_weights (Tensor): The weights of the edges.
        n_obj (Tensor): The number of objects in each scene.
        max_obj (int): The maximum number of objects.
        conn (int): The connectivity parameter for pruning.

        Returns:
        Torch Tensor: (batch_size, max_obj, conn) pruned edge indices.
        """
        pass


class KNNPruning(PruningMethod):
    def prune(self, edge_weights, n_obj, max_obj, conn, foreground_mask):
        batch_size = edge_weights.size(0)
        pruned_edges = torch.zeros((batch_size, max_obj, conn), dtype=torch.long)
        for i in range(batch_size):
            valid_n = n_obj[i].item()
            weights = edge_weights[i]
            weights[:, ~foreground_mask[i]] = -float('inf')
            weights = weights[:valid_n, :valid_n]
            _, knn_indices = torch.topk(weights, k=conn, dim=-1)
            pruned_edges[i, :valid_n, :] = knn_indices
        return pruned_edges


class MSTPruning(PruningMethod):
    def prune(self, edge_weights, n_obj, max_obj, conn, foreground_mask):
        batch_size = edge_weights.size(0)
        pruned_edges = torch.zeros((batch_size, max_obj, conn), dtype=torch.long)
        for i in range(batch_size):
            valid_n = n_obj[i].item()
            weights = edge_weights[i]
            weights[:, ~foreground_mask[i]] = -float('inf')
            weights = weights[:valid_n, :valid_n].cpu().numpy()

            G = nx.Graph()
            for u in range(valid_n):
                for v in range(u):
                    G.add_edge(u, v, weight=weights[u, v])
            mst = nx.maximum_spanning_tree(G)
            root = np.random.randint(0, valid_n)  # randomly select a root node

            # Find parent of each node using BFS
            parent = {root: -1}
            queue = [root]
            while queue:
                current = queue.pop(0)
                for neighbor in mst.neighbors(current):
                    if neighbor not in parent:
                        parent[neighbor] = current
                        queue.append(neighbor)

            # Each node connects to its parent in the MST
            adj_matrix = np.zeros((valid_n, valid_n), dtype=np.int32)
            for node in range(valid_n):
                p = parent[node]
                if p != -1:
                    adj_matrix[node, p] = 1
            
            for u in range(valid_n):
                neighbors = np.where(adj_matrix[u] > 0)[0]

                # If neighbors are less than conn, add highest weight edges until conn is met
                if len(neighbors) < conn:
                    # 1. Find all unconnected nodes for the current node
                    # (excluding itself and already connected neighbors)
                    connected = set(neighbors)
                    connected.add(u)
                    candidate_nodes = [v for v in range(valid_n) if v not in connected]
                    # 2. Sort candidate nodes by edge weight in descending order
                    if candidate_nodes:
                        # Extract weights from the current node to candidate nodes
                        candidate_weights = [(v, weights[u, v]) for v in candidate_nodes]
                        candidate_weights.sort(key=lambda x: x[1], reverse=True)
                        # 3. Add neighbors until conn is met
                        need = conn - len(neighbors)
                        add_neighbors = [v for v, _ in candidate_weights[:need]]
                        neighbors = np.concatenate([neighbors, add_neighbors])
                # If neighbors exceed conn (theoretically shouldn't happen in MST)
                elif len(neighbors) > conn:
                    neighbors = neighbors[:conn]
                
                # Store in the result tensor
                pruned_edges[i, u, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)
        return pruned_edges