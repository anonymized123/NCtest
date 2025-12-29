import sys
sys.path.append('..')

import pandas as pd
import torch.nn.functional as F
import torch
import argparse

from sklearn.linear_model import LogisticRegression

from utils import *
import torch.utils.data as Data

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.stats import entropy

best_xgb_params = {
    'citeseer': {
        'eta': 0.05,
        'max_depth': 8,
        'gamma': 0.3,
        'subsample': 0.5
    },
    'cora': {
        'eta': 0.2,
    },
    'pubmed': {
        'eta': 0.1,
        'subsample': 0.5,
        'min_child_weight': 2
    },
    'coauthorcs': {
    },
    
}
def get_best_params(dataset_name):
    return best_xgb_params.get(dataset_name, {})

def main(dataset_name, model_name, seed):
    target_model_path = '../GNN_train/models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
    path_x_np = '../data/{}/x_np.pkl'.format(dataset_name)
    
    # path_edge_index = '../data/{}/edge_index_np.pkl'.format(dataset_name)
    # path_edge_index = '../data/{}/attack/0.3_minmax.pkl'.format(dataset_name)
    path_edge_index = '../data/{}/attack/0.3_pgdattack.pkl'.format(dataset_name)
    
    path_y = '../data/{}/y_np.pkl'.format(dataset_name)
    subject_name = '{}_{}'.format(dataset_name, model_name)

    target_hidden_channel = 16
    # path_result_pfd = 'results/pfd' + '_' + subject_name + '.csv'
    # path_result_apfd = 'results/apfd' + '_' + subject_name + '.csv'

    ################# load data
    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    num_node_features = len(x[0])
    num_classes = len(set(y))
    num_node = x.shape[0]
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)
    #################

    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)

    ##########################################
    path_degAttr = './attributes/DEG/degAttr_{}_{}.npy'.format(dataset_name, model_name)
    path_deterAttr = './attributes/DETER/{}/deterAttr_{}_{}.npy'.format(seed, dataset_name, model_name)
    path_mlpAttr = './attributes/MLP/{}/mlpAttr_{}_{}.npy'.format(seed,dataset_name, model_name)
    path_probaAttr = './attributes/PRO/{}/probaAttr_{}_{}.npy'.format(seed,dataset_name, model_name)

    deterAttr = np.load(path_deterAttr)
    probaAttr = np.load(path_probaAttr)
    mlpAttr = np.load(path_mlpAttr)
    degAttr = np.load(path_degAttr)
    # print(deterAttr.shape, probaAttr.shape, mlpAttr.shape, degAttr.shape)
    Attr1 = np.concatenate((deterAttr, probaAttr, mlpAttr, degAttr), axis=1)
    # Attr_final = Attr1
    ##########################################
    Attr_agg = np.copy(Attr1)
    num_nei = np.ones(shape=num_node)

    for i in tqdm(range(edge_index.shape[1])):
        Attr_agg[edge_index[0][i]] += Attr1[edge_index[1][i]]
        num_nei[edge_index[0][i]] += 1
    
    for i in tqdm(range(num_node)):
        if num_nei[i] > 0:
            Attr_agg[i] /= num_nei[i]
        # else:
        #     Attr_agg[i] = Attr1[i]
    Attr_final = np.concatenate((Attr1, Attr_agg), axis=1)
 
    ##########################################
  
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    target_pre_train, target_pre_test = target_pre[train_idx], target_pre[test_idx]
    
    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)
    
    idx_miss_list = get_idx_miss_class(target_pre_test, test_y)

    feature_np = Attr_final 

    x_train = feature_np[train_idx]
    y_train = label_np[train_idx]
    x_test = feature_np[test_idx]
    y_test = label_np[test_idx]

    # XGB
    model = XGBClassifier(**get_best_params(dataset_name))
    model.fit(x_train, y_train)
    xgb_pre = model.predict(x_test)
    y_pred_train_xgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_xgb = model.predict_proba(x_test)[:, 1]
    xgb_rank_idx = y_pred_test_xgb.argsort()[::-1].copy()
    xgb_rank_idx_train = y_pred_train_xgb.argsort()[::-1].copy()
    
    # LR
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)
    lr_pre = model.predict(x_test)
    y_pred_train_lr = model.predict_proba(x_train)[:, 1]
    y_pred_test_lr = model.predict_proba(x_test)[:, 1]
    lr_rank_idx = y_pred_test_lr.argsort()[::-1].copy()
    lr_rank_idx_train = y_pred_train_lr.argsort()[::-1].copy()

    # RF
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    rf_pre = model.predict(x_test)
    y_pred_train_rf = model.predict_proba(x_train)[:, 1]
    y_pred_test_rf = model.predict_proba(x_test)[:, 1]
    rf_rank_idx = y_pred_test_rf.argsort()[::-1].copy()
    rf_rank_idx_train = y_pred_train_rf.argsort()[::-1].copy()

    # LGBM
    model = LGBMClassifier()
    model.fit(x_train, y_train)
    lgb_pre = model.predict(x_test)
    y_pred_train_lgb = model.predict_proba(x_train)[:, 1]
    y_pred_test_lgb = model.predict_proba(x_test)[:, 1]
    lgb_rank_idx = y_pred_test_lgb.argsort()[::-1].copy()
    lgb_rank_idx_train = y_pred_train_lgb.argsort()[::-1].copy()


    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    lr_apfd = apfd(idx_miss_list, lr_rank_idx)
    rf_apfd = apfd(idx_miss_list, rf_rank_idx)
    xgb_apfd = apfd(idx_miss_list, xgb_rank_idx)
    
    dropout_rank_idx = probaAttr[test_idx][:, 0].argsort()[::-1].copy()
    dropout_apfd = apfd(idx_miss_list, dropout_rank_idx)
    
    select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    xgb_ratio_list = get_res_ratio_list(idx_miss_list, xgb_rank_idx, select_ratio_list)
    lgb_ratio_list = get_res_ratio_list(idx_miss_list, lgb_rank_idx, select_ratio_list)
    lr_ratio_list = get_res_ratio_list(idx_miss_list, lr_rank_idx, select_ratio_list)
    rf_ratio_list = get_res_ratio_list(idx_miss_list, rf_rank_idx, select_ratio_list)
    dropout_ratio_list = get_res_ratio_list(idx_miss_list, dropout_rank_idx, select_ratio_list)
    
    rank_idx_dict = {
        'dropout': dropout_rank_idx,
        'gr_xgb': xgb_rank_idx,    
        'gr_lgb': lgb_rank_idx,
        'gr_lr': lr_rank_idx,
        'gr_rf': rf_rank_idx
    }
    with open('result/GraphRank_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'wb') as f:
        pickle.dump(rank_idx_dict, f)
        
    return [dropout_apfd, xgb_apfd, lgb_apfd, lr_apfd, rf_apfd], [dropout_ratio_list, xgb_ratio_list, lgb_ratio_list, lr_ratio_list, rf_ratio_list]

    # return [dropout_apfd, xgb_apfd, xgb_apfd, xgb_apfd, xgb_apfd], [dropout_ratio_list, xgb_ratio_list, xgb_ratio_list, xgb_ratio_list, xgb_ratio_list]

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
    
    apfd_list, pfdn_list  = main(dataset_name, model_name, seed)
    columns = ['dropout_apfd', 'xgb_apfd', 'lgb_apfd', 'lr_apfd', 'rf_apfd']
    df_apfd = pd.DataFrame(data=[apfd_list], columns=columns)
    df_pfdn = pd.DataFrame(data=[pfdn_list], columns=columns)

    if model_name == 'gat' and dataset_name == 'citeseer':
        df_apfd.to_csv('result/GraphRank_{}.csv'.format(seed), mode='w', header=True, index=False)
    else:
        df_apfd.to_csv('result/GraphRank_{}.csv'.format(seed), mode='a', header=False, index=False)
    
    if model_name == 'gat' and dataset_name == 'citeseer':
        df_pfdn.to_csv('result/GraphRank_pfdn_{}.csv'.format(seed), mode='w', header=True, index=False)
        
    else:
        df_pfdn.to_csv('result/GraphRank_pfdn_{}.csv'.format(seed), mode='a', header=False, index=False)