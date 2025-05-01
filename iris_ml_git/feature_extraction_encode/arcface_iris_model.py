# Define a dual-branch deep learning architecture for iris recognition
# The backbone network is EfficientNet-B0, used for extracting robust iris features
# Two output heads:
#   (1) An ArcFace classification head, which introduces angular margin to enhance
#       inter-class separability and improve recognition accuracy.
#   (2) A hash encoding head that projects features into compact binary codes,
#       supporting efficient matching via Hamming distance.

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class L2Norm(nn.Module):
    # L2 normalize feature vectors
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class ArcMarginProduct(nn.Module):
    # ArcFace head: adds angular margin to improve separability
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        phi = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return self.s * output

class ArcFaceIrisNet(nn.Module):
    # Iris recognition model: EfficientNet backbone + ArcFace classification + hash encoding
    def __init__(self, num_classes, hash_bits=128):
        super().__init__()
        self.hash_dim = hash_bits
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        self.feature_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.l2norm = L2Norm()
        self.hash_layer = nn.Sequential(
            nn.Linear(self.feature_dim, hash_bits),
            nn.BatchNorm1d(hash_bits)
        )
        self.classifier = ArcMarginProduct(self.feature_dim, num_classes)

    def forward(self, x, label=None):
        x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        norm_feat = self.l2norm(feat)
        hash_code = self.hash_layer(feat)
        if label is not None:
            logits = self.classifier(norm_feat, label)
            return hash_code, logits
        return hash_code, None
