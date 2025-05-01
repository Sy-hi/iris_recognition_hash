# Multi-task loss for iris recognition: ArcFace + Hash + Triplet + Center

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskIrisLoss(nn.Module):
    def __init__(self, lambda_hash=0.01, lambda_triplet=0.1, lambda_center=0.1, num_classes=1000, label_smoothing=0.1):
        super().__init__()
        self.lambda_hash = lambda_hash # weight for hash loss
        self.lambda_triplet = lambda_triplet # weight for triplet loss
        self.lambda_center = lambda_center # weight for center loss
        self.num_classes = num_classes # number of classes
        self.label_smoothing = label_smoothing # label smoothing factor

        self.ce = nn.CrossEntropyLoss()  # default CE
        self.triplet = nn.TripletMarginLoss(margin=1.0) # triplet loss
        self.center_loss = None  # optional external center loss


    def smooth_cross_entropy(self, logits, target):
        # CrossEntropy with label smoothing
        # Apply label smoothing CE
        if self.label_smoothing == 0:
            return self.ce(logits, target)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.label_smoothing)
        log_prob = F.log_softmax(logits, dim=1)
        return -(true_dist * log_prob).sum(dim=1).mean()

    def forward(self, logits, labels, hash_feat, anchor=None, positive=None, negative=None):
        # Compute total multi-task loss
        arc_loss = self.smooth_cross_entropy(logits, labels) # classification loss
        hash_loss = ((hash_feat.abs() - 1) ** 2).mean() # hash regularization
        total = arc_loss + self.lambda_hash * hash_loss

        if anchor is not None and positive is not None and negative is not None:
            triplet_loss = self.triplet(anchor, positive, negative)
            total += self.lambda_triplet * triplet_loss
        else:
            triplet_loss = torch.tensor(0.0, device=logits.device)

        if self.center_loss is not None:
            c_loss = self.center_loss(labels, hash_feat)
            total += self.lambda_center * c_loss
        else:
            c_loss = torch.tensor(0.0, device=logits.device)

        # return total and individual loss components
        return total, arc_loss.item(), hash_loss.item(), triplet_loss.item(), c_loss.item()
