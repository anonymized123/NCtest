import sys
sys.path.append('..')

import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn.functional as F

from sklearn.model_selection import ParameterGrid
from config import *
from utils import set_seeds, select_model
from sklearn.model_selection import train_test_split

def train(hidden_channel, x, y, edge_index, num_node_features, num_classes, train_idx, test_idx, save_model_name, epochs, dic):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = select_model(model_name, hidden_channel, num_node_features, num_classes, dic)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_model_name)

    model.eval()
    pred = model(x, edge_index).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)
    print()

def main(dataset_name, model_name, list_dic, seed):
    path_x_np = '../data/{}/x_np.pkl'.format(dataset_name)
    path_edge_index = '../data/{}/edge_index_np.pkl'.format(dataset_name)
    path_y = '../data/{}/y_np.pkl'.format(dataset_name)
    
    ########### load data
    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    ###########
    
    path_save = './mutation_models/{}_{}/{}/{}_{}_'.format(dataset_name, model_name, seed, dataset_name, model_name)
    
    if not os.path.exists('./mutation_models/{}_{}/{}'.format(dataset_name, model_name, seed)):
        os.makedirs('./mutation_models/{}_{}/{}'.format(dataset_name, model_name, seed), exist_ok=True)
        
    j = 0
    for epochs in epochs_list:
        for i in hidden_channel_list:
            for tmp_dic in list_dic:     
                save_model_name = path_save + str(i) + '_' + str(j) + '.pt'
                pickle.dump(tmp_dic, open(path_save + str(i) + '_' + str(j) + '.pkl', 'wb'))
                train(i, x, y, edge_index, num_node_features, num_classes, train_idx, test_idx, save_model_name, epochs, tmp_dic)
                j += 1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    seed = args.seed 
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    set_seeds(seed)

        
    if model_name == 'gcn':
        epochs_list = epochs_gcn
        dic_mutation = dic_mutation_gcn
    if model_name == 'gat':
        epochs_list = epochs_gat
        dic_mutation = dic_mutation_gat
    if model_name == 'graphsage':
        epochs_list = epochs_graphsage
        dic_mutation = dic_mutation_graphsage
    if model_name == 'tagcn':
        epochs_list = epochs_tagcn
        dic_mutation = dic_mutation_tagcn
    list_dic = list(ParameterGrid(dic_mutation))
    
    if model_name == 'gcn':
        dic_mutation_gcn_1 = {
            "normalize": [True, False],
            "bias": [True],
            "improved": [True, False],
            "cached": [True, False],
            "add_self_loops": [False]
        }
        dic_mutation_gcn_2 = {
            "normalize": [True],
            "bias": [True],
            "improved": [True, False],
            "cached": [True, False],
            "add_self_loops": [True]
        }
        list_dic = list(ParameterGrid(dic_mutation_gcn_1)) + list(ParameterGrid(dic_mutation_gcn_2))     
    
    
    main(dataset_name, model_name, list_dic, seed)