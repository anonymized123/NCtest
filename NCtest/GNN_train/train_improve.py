import sys
sys.path.append('..')
import pandas as pd
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

def load_target_model(model_name, num_node_features, hidden_channel, num_classes, target_model_path):  
    if model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes)
    model.load_state_dict(torch.load(target_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

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


def get_rank_idx(method):
    if method == 'dnn' or method == 'mutation' or method == 'lgb' or method == 'xgb' or method == 'lr' or method == 'rf' or method == 'deepGini' or method == 'leastConfidence' or method == 'margin' or method == 'entropy' or method == 'random':
        with open('../GraphPrior/result/GraphPrior_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'rb') as f:
            rank_idx_dict = pickle.load(f)
            return rank_idx_dict[method]
    elif method == 'stacking' or method == 'voting':
        with open('../NodeRank/result/NodeRank_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'rb') as f:
            rank_idx_dict = pickle.load(f)
            return rank_idx_dict[method]
    
    elif method == 'dropout' or method == 'gr_xgb' or method == 'gr_lgb' or method == 'gr_lr' or method == 'gr_rf':
        with open('../GraphRank/result_全部/GraphRank_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'rb') as f:
            rank_idx_dict = pickle.load(f)
            return rank_idx_dict[method]
    elif method == 'qkv':
        with open('../My/result_attn_base/NCtest_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'rb') as f:
            rank_idx_dict = pickle.load(f)
            return rank_idx_dict[method]
    else:
        raise ValueError('Invalid method name')

def get_ori_acc(x, edge_index, y, num_node_features, hidden_channel, num_classes, path_save_model, device):
    target_model = load_target_model(model_name, num_node_features, hidden_channel, num_classes, path_save_model).to(device)
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)
    
    pred = target_model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    train_acc = int(correct) / len(train_idx)
    print('Original train acc:', train_acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    test_acc = int(correct) / len(test_idx)
    print('Original test acc:', test_acc)
    return train_acc, test_acc

def train(dataset_name, model_name, seed, method):
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


    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)
    rank_idx = get_rank_idx(method)

    top_n_test_node_idx = rank_idx[: int(len(x) / 10)]
    new_node_idx = test_idx[top_n_test_node_idx]
    train_idx = np.concatenate([train_idx, new_node_idx])
    test_idx = np.setdiff1d(test_idx, new_node_idx)


    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)


    device = torch.device('cpu')
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



    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    train_acc = int(correct) / len(train_idx)
    print('train:', train_acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    test_acc = int(correct) / len(test_idx)
    print('test:', test_acc)

    ori_train_acc, ori_test_acc = get_ori_acc(x, edge_index, y, num_node_features, hidden_channel, num_classes, path_save_model, device)
    train_improve = train_acc - ori_train_acc
    test_improve = test_acc - ori_test_acc
    print('train improve:', train_improve)
    print('test improve:', test_improve)
    return train_improve, test_improve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    
    args = parser.parse_args()
    seed = args.seed 
    dataset_name = args.dataset_name
    model_name = args.model_name
    method = args.method
    
    set_seeds(seed)
    train_improve, test_improve = train(dataset_name, model_name, seed, method)
    
    columns = ['train_improve', 'test_improve']
    improve_list = pd.DataFrame(data=[[train_improve, test_improve]], columns=columns)

    if model_name == 'gat' and dataset_name == 'citeseer':
        improve_list.to_csv('model_improve/{}_{}.csv'.format(method, seed), mode='w', header=True, index=False)
        
    else:
        improve_list.to_csv('model_improve/{}_{}.csv'.format(method, seed), mode='a', header=False, index=False)