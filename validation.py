from tasks import predict_crime, lu_classify, predict_popus
import torch
import pickle


if __name__ == '__main__':
    emb = pickle.load(open('./save_emb/emb_1_.pickle', 'rb'))
    predict_crime(emb)
    predict_popus(emb)
    lu_classify(emb)
