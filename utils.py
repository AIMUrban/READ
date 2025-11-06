import torch
import torch.nn.functional as F
import pickle
import numpy as np
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_adj(feature, name: str, save: bool):
    # print(feature.shape)
    adj = torch.zeros(270, 270)
    feature = F.normalize(feature, dim=-1)
    for i in range(270):
        for j in range(270):
            adj[i][j] = F.cosine_similarity(feature[i].view(1, -1), feature[j].view(1, -1))

    if save:
        pickle.dump(adj, open(f'./data/{name}.pickle', 'wb'))
    return adj


def get_mob(flow_matrix):
    mob = flow_matrix / torch.mean(flow_matrix)
    return mob