import sys
sys.path.append('..')

from models import GAT, GCN, GraphSAGE, TAGCN
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils import set_seeds
from sklearn.model_selection import train_test_split
import random
import argparse
import os


def get_model(model_name, num_node_features, hidden_channel, num_classes):
    if model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes)
    return model


def train(dataset_name, model_name, seed):
    # data 路径
    path_x_np = '../data/{}/x_np.pkl'.format(dataset_name)
    path_edge_index = '../data/{}/edge_index_np.pkl'.format(dataset_name)
    path_y = '../data/{}/y_np.pkl'.format(dataset_name)
    

    # 载入data
    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    
    # 特征维度    类别数量
    num_node_features = len(x[0])
    num_classes = len(set(y))

    epochs = 10
    path_save_model = './models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
    hidden_channel = 16
    
    if not os.path.exists('./models_save/{}'.format(seed)):
        os.mkdir('./models_save/{}'.format(seed))

    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, num_node_features, hidden_channel, num_classes)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    # torch.save(model.state_dict(), path_save_model)

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
            
    args = parser.parse_args()
    seed = args.seed 
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    set_seeds(seed)
    train(dataset_name, model_name, seed)