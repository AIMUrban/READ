import torch
from layers import *
from torch_geometric.nn import GCNConv
import numpy as np
import random
import torch.nn.functional as F

seed = 6
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


class READ(nn.Module):
    def __init__(self, hidden_dim, gcn_layers, num_heads=8):
        super().__init__()
        self.region_emb = None
        self.adjs = None
        self.gcn_layers = gcn_layers

        self.attn_se_attr = CityContextTransformerEncoder(num_heads, 9, hidden_dim)
        self.attn_se_inflow = CityContextTransformerEncoder(num_heads, 270, hidden_dim)
        self.attn_se_outflow = CityContextTransformerEncoder(num_heads, 270, hidden_dim)

        self.gcn_convs = nn.ModuleList([GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_dim, affine=True) for _ in range(self.gcn_layers-1)])

        self.get_dynamic_adj_attr = AdaptiveWeightedMultiGraphConstruction(feature_size=hidden_dim)
        self.get_dynamic_adj_inflow = AdaptiveWeightedMultiGraphConstruction(feature_size=hidden_dim)
        self.get_dynamic_adj_outflow = AdaptiveWeightedMultiGraphConstruction(feature_size=hidden_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        alpha = 0.2
        self.alpha_attr = alpha
        self.alpha_inflow = alpha
        self.alpha_outflow = alpha

        self.fused_layer = Fusion(hidden_dim, hidden_dim)

        self.attr_out = None
        self.inflow_out = None
        self.outflow_out = None

        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, view_attr, view_inflow, view_outflow, adj_, mob):
        adj_attr = adj_

        view_attr_f, view_attr_attn = self.attn_se_attr(view_attr)
        adj_attr_ = self.get_dynamic_adj_attr(view_attr_f)
        attr_coo = adj_attr_.to_sparse_coo()

        view_inflow_f, view_inflow_attn = self.attn_se_inflow(view_inflow)
        adj_inflow_ = self.get_dynamic_adj_inflow(view_inflow_f)
        inflow_coo = adj_inflow_.to_sparse_coo()

        view_outflow_f, view_outflow_attn = self.attn_se_outflow(view_outflow)
        adj_outflow_ = self.get_dynamic_adj_outflow(view_outflow_f)
        outflow_coo = adj_outflow_.to_sparse_coo()

        for i in range(self.gcn_layers - 1):
            view_attr_f = self.gcn_convs[i](view_attr_f, attr_coo.indices(), attr_coo.values())
            view_attr_f = self.bns[i](view_attr_f)
            view_attr_f = F.leaky_relu(view_attr_f)

            view_inflow_f = self.gcn_convs[i](view_inflow_f, inflow_coo.indices(), inflow_coo.values())
            view_inflow_f = self.bns[i](view_inflow_f)
            view_inflow_f = F.leaky_relu(view_inflow_f)

            view_outflow_f = self.gcn_convs[i](view_outflow_f, outflow_coo.indices(), outflow_coo.values())
            view_outflow_f = self.bns[i](view_outflow_f)
            view_outflow_f = F.leaky_relu(view_outflow_f)

        view_attr_f = self.gcn_convs[-1](view_attr_f, attr_coo.indices(), attr_coo.values())
        view_inflow_f = self.gcn_convs[-1](view_inflow_f, inflow_coo.indices(), inflow_coo.values())
        view_outflow_f = self.gcn_convs[-1](view_outflow_f, outflow_coo.indices(), outflow_coo.values())

        view_features = torch.stack([view_attr_f, view_inflow_f, view_outflow_f])
        attn_features, _ = self.self_attn(view_features, view_features, view_features)

        out_attr = attn_features[0] * self.alpha_attr + (1 - self.alpha_attr) * view_attr_f
        out_s = attn_features[1] * self.alpha_inflow + (1 - self.alpha_inflow) * view_inflow_f
        out_d = attn_features[2] * self.alpha_outflow + (1 - self.alpha_outflow) * view_outflow_f

        features = torch.stack([out_attr, out_s, out_d])
        fused_features = self.fused_layer(features)

        self.region_emb = fused_features

        beta = 0.5
        out = beta * view_features + (1 - beta) * fused_features

        attr_loss = self.semantic_relation_reconstruction_loss(out[0], adj_attr)
        mob_loss = self.mobility_prediction_loss(out[1], out[2], mob)

        loss = mob_loss + attr_loss
        return loss

    def semantic_relation_reconstruction_loss(self, embeds: torch.Tensor, adj):
        mask = adj != 0
        inner_prod = embeds @ embeds.T
        pred = torch.masked_select(inner_prod, mask)
        value = torch.masked_select(adj, mask)
        return self.mse_loss(pred, value)

    def mobility_prediction_loss(self, s_embeds: torch.Tensor, d_embeds: torch.Tensor, mob):
        mask = mob != 0

        inner_prod_s = torch.matmul(s_embeds, d_embeds.transpose(-2, -1))
        ps_hat = F.log_softmax(inner_prod_s, dim=-1)
        ps_hat = torch.masked_select(ps_hat, mask)
        mob_s = torch.masked_select(mob, mask)

        inner_prod_d = torch.matmul(d_embeds, s_embeds.transpose(-2, -1))
        pd_hat = F.log_softmax(inner_prod_d, dim=-1)
        pd_hat = torch.masked_select(pd_hat, mask.T)
        mob_d = torch.masked_select(mob.T, mask.T)

        loss = torch.sum(-torch.mul(mob_s, ps_hat) - torch.mul(mob_d, pd_hat))
        return loss

