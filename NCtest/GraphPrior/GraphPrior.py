import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import pickle
import numpy as np

from get_rank_idx import *
from utils import set_seeds, get_model_path, load_mutation_model, load_target_model, get_idx_miss_class, get_mutation_model_features
import torch.utils.data as Data
from config import *
from sklearn.linear_model import LogisticRegression
from dnn import DNN, get_acc
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import argparse


def main(dataset_name, model_name, seed):
    path_model_file = './mutation_models/{}_{}/{}'.format(dataset_name, model_name, seed)
    target_model_path = '../GNN_train/models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
    path_x_np = '../data/{}/x_np.pkl'.format(dataset_name)

    # path_edge_index = '../data/{}/edge_index_np.pkl'.format(dataset_name)
    # path_edge_index = '../data/{}/attack/0.3_minmax.pkl'.format(dataset_name)
    path_edge_index = '../data/{}/attack/0.3_pgdattack.pkl'.format(dataset_name)
    
    path_y = '../data/{}/y_np.pkl'.format(dataset_name)
    subject_name = '{}_{}_clean'.format(dataset_name, model_name)
    path_result = 'res/res_misclassification_{}_{}.csv'.format(dataset_name, model_name)
    path_pre_result = 'res/pre_res_misclassification_{}_{}.csv'.format(dataset_name, model_name)
    target_hidden_channel = 16
    
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
    ###########

    
    # 这里把所有 mutation models 弄出来了
    path_model_list = get_model_path(path_model_file)
    path_model_list = sorted(path_model_list)
    path_config_list = [i.replace('.pt', '.pkl') for i in path_model_list]
    hidden_channel_list = [int(i.split('/')[-1].split('_')[2]) for i in path_config_list]
    dic_list = [pickle.load(open(i, 'rb')) for i in path_config_list]

    model_list = []
    for i in range(len(path_model_list)):
        try:
            tmp_model = load_mutation_model(model_name, path_model_list[i], hidden_channel_list[i], num_node_features, num_classes, dic_list[i])
            model_list.append(tmp_model)
        except:
            print(dic_list[i])

    print('number of models:', len(path_model_list))
    print('number of models loaded:', len(model_list))

    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)

    # 生成二分类器的 训练 测试 数据
    feature_np, label_np = get_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, model_list, model_name)
    x_train = feature_np[train_idx]
    y_train = label_np[train_idx]
    x_test = feature_np[test_idx]
    y_test = label_np[test_idx]

    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test.argsort()[::-1].copy()
    lr_pre_list = list(y_pred_test).copy()

    # RF
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    rf_rank_idx = y_pred_test.argsort()[::-1].copy()
    rf_pre_list = list(y_pred_test).copy()

    # LGB
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    lgb_rank_idx = y_pred_test.argsort()[::-1].copy()
    lgb_pre_list = list(y_pred_test).copy()

    # XGB
    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred_test = model.predict_proba(x_test)[:, 1]
    xgb_rank_idx = y_pred_test.argsort()[::-1].copy()
    xgb_pre_list = list(y_pred_test).copy()

    # DNN
    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    x_train_t.to(device='cuda')
    y_train_t.to(device='cuda')

    x_test_t = torch.from_numpy(x_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    x_test_t.to(device='cuda')
    y_test_t.to(device='cuda')

    input_dim = len(feature_np[0])
    hiden_dim = 8
    output_dim = 2
    dataset = Data.TensorDataset(x_train_t, y_train_t)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    model = DNN(input_dim, hiden_dim, output_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fun = nn.CrossEntropyLoss()

    for e in range(20):
        epoch_loss = 0
        epoch_acc = 0
        for i, (x_t, y_t) in enumerate(dataloader):
            optim.zero_grad()
            out = model(x_t)
            loss = loss_fun(out, y_t)
            loss.backward()
            optim.step()
            epoch_loss += loss.data
            epoch_acc += get_acc(out, y_t)

    y_pred_test = model(x_test_t).detach().numpy()[:, 1]
    dnn_rank_idx = y_pred_test.argsort()[::-1].copy()
    dnn_pre_list = list(y_pred_test).copy()

    # 错误分类的节点位置   注意只关心测试集中的   从 0 开始 末尾多了一个数字，其指示测试集大小
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()[test_idx]
    idx_miss_list = get_idx_miss_class(target_pre, test_y)
    #################
    
    # 计算被 kill 的数量排序
    mutation_rank_idx = Mutation_rank_idx(num_node_features, target_hidden_channel, num_classes, target_model_path, x,
                                          edge_index, test_idx, model_list, model_name)
    ######################
    
    
    x_test_target_model_pre = target_model(x, edge_index).detach().numpy()[test_idx]
    
    margin_rank_idx = Margin_rank_idx(x_test_target_model_pre)
    deepGini_rank_idx = DeepGini_rank_idx(x_test_target_model_pre)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(x_test_target_model_pre)
    entropy_rank_idx = Entropy_rank_idx(x_test_target_model_pre)
    random_rank_idx = Random_rank_idx(x_test_target_model_pre)
    
    # 计算每 10% 找出多少比例的错误
    dnn_ratio_list = get_res_ratio_list(idx_miss_list, dnn_rank_idx, select_ratio_list)
    lgb_ratio_list = get_res_ratio_list(idx_miss_list, lgb_rank_idx, select_ratio_list)
    xgb_ratio_list = get_res_ratio_list(idx_miss_list, xgb_rank_idx, select_ratio_list)
    rf_ratio_list = get_res_ratio_list(idx_miss_list, rf_rank_idx, select_ratio_list)
    lr_ratio_list = get_res_ratio_list(idx_miss_list, lr_rank_idx, select_ratio_list)
    mutation_ratio_list = get_res_ratio_list(idx_miss_list, mutation_rank_idx, select_ratio_list)
    margin_ratio_list = get_res_ratio_list(idx_miss_list, margin_rank_idx, select_ratio_list)
    deepGini_ratio_list = get_res_ratio_list(idx_miss_list, deepGini_rank_idx, select_ratio_list)
    leastConfidence_ratio_list = get_res_ratio_list(idx_miss_list, leastConfidence_rank_idx, select_ratio_list)
    random_ratio_list = get_res_ratio_list(idx_miss_list, random_rank_idx, select_ratio_list)
    entropy_ratio_list = get_res_ratio_list(idx_miss_list, entropy_rank_idx, select_ratio_list)
    
   
    ## 计算APFD
    dnn_apfd = apfd(idx_miss_list, dnn_rank_idx)
    mutation_apfd = apfd(idx_miss_list, mutation_rank_idx)
    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    lr_apfd = apfd(idx_miss_list, lr_rank_idx)
    rf_apfd = apfd(idx_miss_list, rf_rank_idx)
    xgb_apfd = apfd(idx_miss_list, xgb_rank_idx)
    
    deepGini_apfd = apfd(idx_miss_list, deepGini_rank_idx)
    leastConfidence_apfd = apfd(idx_miss_list, leastConfidence_rank_idx)
    margin_apfd = apfd(idx_miss_list, margin_rank_idx)
    entropy_apfd = apfd(idx_miss_list, entropy_rank_idx)
    random_apfd = apfd(idx_miss_list, random_rank_idx)
    
    rank_idx_dict = {
        'dnn': dnn_rank_idx,
        'mutation': mutation_rank_idx,
        'lgb': lgb_rank_idx,
        'xgb': xgb_rank_idx,
        'lr': lr_rank_idx,
        'rf': rf_rank_idx,
        'deepGini': deepGini_rank_idx,
        'leastConfidence': leastConfidence_rank_idx,
        'margin': margin_rank_idx,
        'entropy': entropy_rank_idx,
        'random': random_rank_idx,
    }
    with open('result/GraphPrior_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'wb') as f:
        pickle.dump(rank_idx_dict, f)
    
    return [dnn_apfd, mutation_apfd, lgb_apfd, xgb_apfd, lr_apfd, rf_apfd, deepGini_apfd, leastConfidence_apfd, margin_apfd, entropy_apfd, random_apfd], \
        [dnn_ratio_list, mutation_ratio_list, lgb_ratio_list, xgb_ratio_list, lr_ratio_list, rf_ratio_list, deepGini_ratio_list, leastConfidence_ratio_list, margin_ratio_list, entropy_ratio_list, random_ratio_list]


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
    
    apfd_list, pfdn_list = main(dataset_name, model_name, seed)
    columns = ['dnn_apfd', 'mutation_apfd', 'lgb_apfd', 'xgb_apfd', 'lr_apfd', 'rf_apfd', 'deepGini_apfd', 'leastConfidence_apfd', 'margin_apfd', 'entropy_apfd', 'random_apfd']
    df_apfd = pd.DataFrame(data=[apfd_list], columns=columns)
    df_pfdn = pd.DataFrame(data=[pfdn_list], columns=columns)

    if model_name == 'gat' and dataset_name == 'citeseer':
        df_apfd.to_csv('result/GraphPrior_{}.csv'.format(seed), mode='w', header=True, index=False)
        
    else:
        df_apfd.to_csv('result/GraphPrior_{}.csv'.format(seed), mode='a', header=False, index=False)
        
    if model_name == 'gat' and dataset_name == 'citeseer':
        df_pfdn.to_csv('result/GraphPrior_pfdn_{}.csv'.format(seed), mode='w', header=True, index=False)
        
    else:
        df_pfdn.to_csv('result/GraphPrior_pfdn_{}.csv'.format(seed), mode='a', header=False, index=False)