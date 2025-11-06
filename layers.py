import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

seed = 6
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


class CityContextTransformerEncoder(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, head_num, real_emb_size, feature_size):
        super().__init__()
        assert feature_size % head_num == 0
        self.head_num = head_num

        self.raw_embedding = nn.Sequential(
            nn.Linear(real_emb_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU()
        )

        self.WQ = nn.Linear(feature_size, feature_size)
        self.WK = nn.Linear(feature_size, feature_size)
        self.WV = nn.Linear(feature_size, feature_size)

        self.ln = nn.BatchNorm1d(feature_size, affine=True)
        self.output_linear = nn.Linear(feature_size, feature_size)
        self.attn = nn.MultiheadAttention(embed_dim=feature_size, num_heads=head_num)

    def forward(self, real_emb):
        # region_num, feature_size = input_features.shape
        features = self.raw_embedding(real_emb)

        # region_num = query.size(1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.WQ(features)
        key = self.WK(features)
        value = self.WV(features)

        # 2) Apply attention on all the projected vectors in batch.
        sp, attn = self.attn(query, key, value)
        x_ = (value - sp) / torch.sqrt(torch.matmul(attn, (value - sp) ** 2))

        out = self.output_linear(self.ln(x_)) + x_
        return out, attn


class AdaptiveWeightedMultiGraphConstruction(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.adj_layer_ = nn.Sequential(
            nn.Linear(2 * feature_size, feature_size),
            nn.Tanh(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )

        self.adj_attn_layer = nn.Sequential(
            torch.nn.Linear(feature_size, 64, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1, bias=False)
        )

    def forward(self, features):
        region_num, feature_size = features.shape

        adj_i = torch.repeat_interleave(features.unsqueeze(0), repeats=region_num, dim=1).view(region_num, -1,
                                                                                               feature_size)
        adj_j = torch.repeat_interleave(features.unsqueeze(0), repeats=region_num, dim=0).view(region_num, -1,
                                                                                               feature_size)

        adj_dynamic = torch.cosine_similarity(adj_i, adj_j, dim=-1)
        adj_out = torch.clamp(adj_dynamic, 0)

        return adj_out


class Fusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight_layer = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        weight = F.leaky_relu(self.weight_layer(features))
        weight = torch.softmax(weight, dim=0)
        view_features = torch.mul(features, weight)
        region_feature = torch.sum(view_features, dim=0, keepdim=True)

        return region_feature
