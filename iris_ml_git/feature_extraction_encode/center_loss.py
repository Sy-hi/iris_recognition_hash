# Center Loss: minimizes intra-class feature distance to class centers

import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device=torch.device("cpu")):
        # Initialize center vector for each class
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        # Compute squared distance to corresponding class center
        batch_size = features.size(0)
        # Select centers by label
        centers_batch = self.centers.index_select(0, labels)
        # Calculate Mean loss over batch
        return ((features - centers_batch) ** 2).sum() / batch_size
