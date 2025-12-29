import sys
sys.path.append('..')

import argparse
import pandas as pd
import torch.nn.functional as F
import torch


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import *
import torch.utils.data as Data
from torch import nn
from tqdm import tqdm
from collections import defaultdict

import xgboost as xgb



from gnn_liner_model import get_gat_liner, get_gcn_liner, get_graphsage_liner, get_tagcn_liner
from gnn_liner_model import get_gcn_Structure

from gnn_L1_model import get_gat_L1, get_gcn_L1, get_graphsage_L1, get_tagcn_L1
from gnn_L2_model import get_gat_L2, get_gcn_L2, get_graphsage_L2, get_tagcn_L2

import torch
from torch import nn, optim
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: raw logits, shape [batch, num_classes]
        logp = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logp)
        # 选中真实类别概率
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1))
        # focal loss核心
        focal = (1 - pt) ** self.gamma * logp
        # alpha支持
        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, device=inputs.device)
            focal = focal * alpha.unsqueeze(0)
        loss = -(targets_onehot * focal).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class NeighborTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_classes, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        # 输入投影
        self.input_fc = nn.Linear(input_dim, d_model)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, input_dim]
        # mask: [batch, seq_len]  (1有效, 0无效)
        x_proj = self.input_fc(x)  # (batch, seq_len, d_model)

        # Transformer编码 (batch_first=True)
        x_enc = self.transformer(x_proj, src_key_padding_mask=mask)

        # 常见做法：取CLS token（即序列第1个，feature_th[:,0]，自己的特征）
        x_cls = x_enc[:, 0, :]  # (batch, d_model)

        out = self.fc(x_cls)
        return out


class ModelWithScale(nn.Module):
    def __init__(self, num_classes):
        super(ModelWithScale, self).__init__()
        self.transfor = nn.Linear(num_classes, num_classes)

    def forward(self, logits):
        return self.transfor(logits)


def get_liner(dataset_name, model_name, seed, num_node_features, num_classes):
    if model_name == 'gat':
        return get_gat_liner(dataset_name, model_name, seed, num_node_features, num_classes)
    if model_name == 'gcn':
        return get_gcn_liner(dataset_name, model_name, seed, num_node_features, num_classes)
    if model_name == 'graphsage':
        return get_graphsage_liner(dataset_name, model_name, seed, num_node_features, num_classes)
    if model_name == 'tagcn':
        return get_tagcn_liner(dataset_name, model_name, seed, num_node_features, num_classes)
    print("load liner error !!!!!!!!!\n")
    exit()
    
def main(dataset_name, model_name, seed):
    target_model_path = '../GNN_train/models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
    
    path_x_np = '../data/{}/x_np.pkl'.format(dataset_name)
    
    path_edge_index = '../data/{}/edge_index_np.pkl'.format(dataset_name)
    # path_edge_index = '../data/{}/attack/0.3_minmax.pkl'.format(dataset_name)
    # path_edge_index = '../data/{}/attack/0.3_pgdattack.pkl'.format(dataset_name)
    
    path_y = '../data/{}/y_np.pkl'.format(dataset_name)
    subject_name = '{}_{}'.format(dataset_name, model_name)

    target_hidden_channel = 16

    ################# load data
    x = pickle.load(open(path_x_np, 'rb'))
    edge_index = pickle.load(open(path_edge_index, 'rb'))
    y = pickle.load(open(path_y, 'rb'))

    x_t1 = np.ones_like(x)
  
    num_node_features = len(x[0])
    num_classes = len(set(y))
    num_node = x.shape[0]
    
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)
    
    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    y = torch.from_numpy(y)
    
    x_t1 = torch.from_numpy(x_t1)
    
    #################
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)

    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    target_pre_train, target_pre_test = target_pre[train_idx], target_pre[test_idx]
    
    target_probability = np.exp(target_model(x, edge_index).detach().numpy())
    target_probability_train, target_probability_test = target_probability[train_idx], target_probability[test_idx]
    
    #################
    # Scaling
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # T_model = ModelWithScale(num_classes)
    # T_model.to(device)
    # nll_criterion = nn.CrossEntropyLoss()

    # logits_T = target_model.get_Layer2(x, edge_index)[train_idx]
    # labels_T = y[train_idx]
    # logits_T, labels_T = logits_T.to(device), labels_T.to(device)


    # before_temperature_nll = nll_criterion(logits_T, labels_T).item()
    # print('Before - NLL: %.3f' % (before_temperature_nll))

    # optimizer = optim.LBFGS(T_model.parameters(), lr=0.05, max_iter=200)

    # def closure():
    #     optimizer.zero_grad()
    #     T_out = T_model(logits_T)
    #     loss = nll_criterion(T_out, labels_T)
    #     loss.backward(retain_graph=True)
    #     return loss
    # optimizer.step(closure)

    # after_temperature_nll = nll_criterion(T_model(logits_T), labels_T).item()
    # print('After NLL: %.3f' % (after_temperature_nll))
    # T_model.to('cpu').eval()
    
    #################
    
    L1 = target_model.get_Layer1(x, edge_index).detach().numpy()
    L2 = target_model.get_Layer2(x, edge_index).detach().numpy()
    L3 = target_model(x, edge_index).detach().numpy()   # 这个没有经过 exp ，不是标准的概率分布
    

    
    pred = target_model(x, edge_index).argmax(dim=1).detach().numpy()
    pred_value, _ = torch.max(target_model(x, edge_index), dim=1)
    pred_value = pred_value.detach().numpy()
    
    entropy_score = -np.sum(L3 * np.exp(L3), axis=1)              # entropy

    output_sort = np.sort(np.exp(L3))                               # Margin
    margin_score =  1 - (output_sort[:, -1] - output_sort[:, -2])  # Margin
    
    gini_score = 1 - np.sum(np.power(np.exp(L3), 2), axis=1)   # Gini
    
    select_score = margin_score.copy()
    
    #####################################################
    # cal
    # T_logSoft = F.log_softmax(T_model(target_model.get_Layer2(x, edge_index)), dim=1).detach().numpy()
    # entropy_score_cal = -np.sum(T_logSoft * np.exp(T_logSoft), axis=1)             # entropy

    # output_sort_cal = np.sort(np.exp(T_logSoft))                               # Margin
    # margin_score_cal =  1 - (output_sort_cal[:, -1] - output_sort_cal[:, -2])  # Margin
    
    # gini_score_cal = 1 - np.sum(np.power(np.exp(T_logSoft), 2), axis=1)   # Gini
            
    # select_score = margin_score.copy()
    # select_score_cal = margin_score_cal.copy()
    
    #####################################################

    liner_model = get_liner(dataset_name, model_name, seed, num_node_features, num_classes)
    liner_out = liner_model(x).detach().numpy()
    
    xt1_out = target_model.get_Layer2(x_t1, edge_index).detach().numpy()

    
    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)
    
    idx_miss_list = get_idx_miss_class(target_pre_test, test_y)
    idx_miss_list_train = get_idx_miss_class(target_pre_train, train_y)
    
    # Step 1: 邻接表
    num_nodes = pred.shape[0]
    neighbors = defaultdict(list)
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    for src, dst in zip(src_nodes, dst_nodes):
        neighbors[src.item()].append(dst.item())

    # Step 2: 统计每个节点邻居中“标签不同”比例
    # dif_label_ratio = np.zeros(shape=(num_nodes,1))
    # for i in range(num_nodes):
    #     neigh = neighbors[i]

    #     if not neigh:
    #         dif_label_ratio[i] = 0
    #         continue
    #     label_i = pred[i]
    #     dif_label_count = sum(pred[j] != label_i for j in neigh)
    #     dif_label_ratio[i] = dif_label_count / len(neigh)
        
    nei_label = torch.zeros(size=(num_nodes, num_classes))
    # nei_label = np.zeros(shape=(num_nodes, num_classes))
    
    max_nei_num = 0
    for i in range(num_nodes):
        neigh = neighbors[i]
        max_nei_num = max(max_nei_num, len(neigh))
        for idx in neigh:
            nei_label[i][pred[idx]] += 1
    print("max_nei_num:", max_nei_num)
    # nei_label_min = np.min(nei_label, axis=1, keepdims=True)
    # nei_label_max = np.max(nei_label, axis=1, keepdims=True)
    # denom = nei_label_max - nei_label_min
    # denom[denom == 0] = 1

    # nei_label = (nei_label - nei_label_min) / denom
    nei_label = torch.softmax(nei_label, dim=-1).detach().numpy()

    temp = nei_label
    nei_label = np.zeros(shape=(num_nodes, 1))
    for i in range(num_nodes):
        nei_label[i] = temp[i][pred[i]]
        
    ##########################################
    path_degAttr = '../GraphRank/attributes/DEG/degAttr_{}_{}.npy'.format(dataset_name, model_name)
    path_deterAttr = '../GraphRank/attributes/DETER/{}/deterAttr_{}_{}.npy'.format(seed, dataset_name, model_name)
    path_mlpAttr = '../GraphRank/attributes/MLP/{}/mlpAttr_{}_{}.npy'.format(seed,dataset_name, model_name)
    path_probaAttr = '../GraphRank/attributes/PRO/{}/probaAttr_{}_{}.npy'.format(seed,dataset_name, model_name)

    deterAttr = np.load(path_deterAttr)
    probaAttr = np.load(path_probaAttr)
    mlpAttr = np.load(path_mlpAttr)
    degAttr = np.load(path_degAttr)

    ##########################################
    
    
    feature_np = np.concatenate((L1, L2, liner_out , nei_label, deterAttr, probaAttr, mlpAttr, degAttr), axis=1)
    
    
    # Min-max归一化（按列, 每个特征维度独立处理）
    min_vals = feature_np.min(axis=0, keepdims=True)
    max_vals = feature_np.max(axis=0, keepdims=True)
    denom = max_vals - min_vals
    denom[denom == 0] = 1  # 防止分母为0
    feature_np = (feature_np - min_vals) / denom


    feature_th = torch.zeros(size=(num_nodes, max_nei_num+1, feature_np.shape[1]))
    mask_seq = torch.zeros((num_nodes, max_nei_num+1), dtype=torch.bool)
    for i in range(num_nodes):
        feature_th[i][0] = torch.from_numpy(feature_np[i])
        neigh = neighbors[i]
        num_neigh = len(neigh)
        # mask_seq[i][:num_neigh+1] = False
        # mask_seq[i][num_neigh+1:] = True
        mask_seq[i][1:] = True
        
        for j in range(num_neigh):
            feature_th[i][j+1] = torch.from_numpy(feature_np[neigh[j]])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if dataset_name == 'citeseer':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.1
    #     nhead = 1
    #     epochs = 20
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'cora':
    #     d_model = 80
    #     num_layers = 1
    #     dropout = 0.0
    #     nhead = 1
    #     epochs = 20
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'pubmed':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.0
    #     nhead = 1
    #     epochs = 30
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'coauthorcs':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.1
    #     nhead = 1
    #     epochs = 70
    #     lr = 0.0001
    #     batch_size = 32
    # else:
    #     print("dataset_name error !!!!!!!!")
    #     exit()
        
        
    # mbf_sbf no stru-aware
    if dataset_name == 'citeseer':
        d_model = 64
        num_layers = 1
        dropout = 0.1
        nhead = 1
        epochs = 10
        lr = 0.0005
        batch_size = 32
    elif dataset_name == 'cora':
        d_model = 80
        num_layers = 1
        dropout = 0.0
        nhead = 1
        epochs = 10
        lr = 0.0005
        batch_size = 32
    elif dataset_name == 'pubmed':
        d_model = 64
        num_layers = 1
        dropout = 0.0
        nhead = 1
        epochs = 20
        lr = 0.0005
        batch_size = 32
    elif dataset_name == 'coauthorcs':
        d_model = 64
        num_layers = 1
        dropout = 0.1
        nhead = 1
        epochs = 60
        lr = 0.0001
        batch_size = 32
    else:
        print("dataset_name error !!!!!!!!")
        exit()   
     
     
    # # mbf no stru-aware
    # if dataset_name == 'citeseer':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.5
    #     nhead = 1
    #     epochs = 10
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'cora':
    #     d_model = 80
    #     num_layers = 1
    #     dropout = 0.5
    #     nhead = 1
    #     epochs = 10
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'pubmed':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.5
    #     nhead = 1
    #     epochs = 20
    #     lr = 0.0005
    #     batch_size = 32
    # elif dataset_name == 'coauthorcs':
    #     d_model = 64
    #     num_layers = 1
    #     dropout = 0.5 
    #     nhead = 1
    #     epochs = 20
    #     lr = 0.0001
    #     batch_size = 32
    # else:
    #     print("dataset_name error !!!!!!!!")
    #     exit()       
        
    model = NeighborTransformer(
        input_dim=feature_th.shape[-1],  # 对应 feature 向量长度
        d_model=d_model,                     # 可以自定义
        num_classes=2,                   # 分类数
        num_layers=num_layers,
        dropout=dropout,
        nhead=nhead
    ).to(device)

    # 用训练集索引，统计类别分布
    # labels_train = torch.from_numpy(label_np[train_idx]).to(device)
    # num_class0 = (labels_train == 0).sum().item()
    # num_class1 = (labels_train == 1).sum().item()
    # total = len(labels_train)
    # weight = torch.tensor([
    #     total / (2 * num_class0),
    #     total / (2 * num_class1)
    # ], dtype=torch.float, device=labels_train.device)

    criterion = torch.nn.CrossEntropyLoss(weight=None)
    # criterion = FocalLoss(gamma=2.0, alpha=0.5, reduction='mean')
    
    x_th = feature_th.to(device)
    mask_th = mask_seq.to(device)
    y_th = torch.from_numpy(label_np).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset=train_idx, batch_size=batch_size, shuffle=True)
    model.train()
    loss_list = []
    for epoch in tqdm(range(epochs)):
        loss_temp = 0
        for batch_idx in loader:
            batch_x = x_th[batch_idx]
            batch_mask = mask_th[batch_idx]
            batch_y = y_th[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_x, mask=batch_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
            
        loss_list.append(loss_temp)
    # print("Training loss:", loss_list)
    model.eval()
    with torch.no_grad():
        logits = model(x_th[test_idx], mask=mask_th[test_idx])
        qkv_pre = F.softmax(logits, dim=1)

    qkv_rank_idx = qkv_pre[:, 1].argsort(descending=True).cpu().numpy()
    qkv_apfd = apfd(idx_miss_list, qkv_rank_idx)
    
    
    
    select_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    qkv_ratio_list = get_res_ratio_list(idx_miss_list, qkv_rank_idx, select_ratio_list)
    
    rank_idx_dict = {
        'qkv': qkv_rank_idx,
    }
    with open('result_attn/NCtest_rank_idx_{}_{}_{}.pkl'.format(dataset_name, model_name, seed), 'wb') as f:
        pickle.dump(rank_idx_dict, f)
        
    return [qkv_apfd], [qkv_ratio_list]

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
    
    print(dataset_name, model_name)
    print('###############################')
    
    apfd_list, pfdn_list = main(dataset_name, model_name, seed)


    columns = ['qkv_apfd']
    # columns = np.arange(len(apfd_list))
    df_apfd = pd.DataFrame(data=[apfd_list], columns=columns)
    df_pfdn = pd.DataFrame(data=[pfdn_list], columns=columns)

    if model_name == 'gat' and dataset_name == 'citeseer':
        df_apfd.to_csv('result_attn/apfd_{}.csv'.format(seed), mode='w', header=True, index=False)
        
    else:
        df_apfd.to_csv('result_attn/apfd_{}.csv'.format(seed), mode='a', header=False, index=False)


    if model_name == 'gat' and dataset_name == 'citeseer':
        df_pfdn.to_csv('result_attn/pfdn_{}.csv'.format(seed), mode='w', header=True, index=False)
        
    else:
        df_pfdn.to_csv('result_attn/pfdn_{}.csv'.format(seed), mode='a', header=False, index=False)