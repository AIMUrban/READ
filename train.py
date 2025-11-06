from READ import *
from tasks import predict_crime, lu_classify, predict_popus
import pickle
import torch
from utils import get_adj, get_mob, setup_seed
# setup_seed()

if __name__ == '__main__':
    # set some parameters
    epochs = 1000
    d_feature = 128

    mob = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and preprocess datas
    region_attr = torch.tensor(pickle.load(open('data/region_attribute.pickle', 'rb')), dtype=torch.float32)
    views_attr = region_attr.to(device)
    adj_attr = get_adj(region_attr.squeeze(), 'adj_attr', save=False).to(device)

    region_flow = pickle.load(open('data/region_flow.pickle', 'rb'))
    flow_matrix = torch.tensor(region_flow['flow_matrix'], dtype=torch.float32).squeeze()
    flow_in = torch.tensor(region_flow['inflow_matrix'], dtype=torch.float32)
    flow_out = torch.tensor(region_flow['outflow_matrix'], dtype=torch.float32)

    s_matrix = (flow_matrix / torch.sum(flow_matrix, dim=-1)).to(device)
    d_matrix = (flow_matrix.T / torch.sum(flow_matrix.T, dim=-1)).to(device)

    mob = get_mob(flow_matrix).to(device)

    model = READ(d_feature, gcn_layers=2, num_heads=8).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    results = []
    for epoch in range(epochs):
        loss = model(views_attr, s_matrix, d_matrix, adj_attr, mob)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (epoch+1) % 100 == 0:
            print("Train loss {:.3f} at epoch {}.".format(loss.item(), epoch+1))

            emb = model.region_emb.detach().cpu().numpy().squeeze()
            pickle.dump(emb, open(f'./save_emb/emb.pickle', 'wb')) # if need to save embeddings
            predict_crime(emb)
            predict_popus(emb)
            lu_classify(emb)



