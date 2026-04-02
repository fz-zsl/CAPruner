from abc import ABC, abstractmethod
import torch

class NodeDistance(ABC):
    @abstractmethod
    def calc_dist(self, bboxes):
        """
        Compute node distances
        Args:
            bboxes: (batch_size, num_nodes, 6) node bounding boxes
        Returns:
            dist: (batch_size, num_nodes, num_nodes) node distances
        """
        pass


class LpDistance(NodeDistance):
    def __init__(self, p=2.0):
        """
        Args:
            p: the parameter for p-norm aggregation
        """
        self.p = p

    def calc_dist(self, bboxes):
        """
        Compute Lp distance between node bounding boxes
        Args:
            bboxes: (batch_size, num_nodes, 6) node bounding boxes
        Returns:
            dist: (batch_size, num_nodes, num_nodes) Lp distances
        """
        centers = bboxes[..., :3] + bboxes[..., 3:6]
        dist_matrix = torch.cdist(centers, centers, p=self.p)  # (batch_size, num_nodes, num_nodes)
        return dist_matrix
