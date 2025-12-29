
import os
import random
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from models import GAT, GCN, GraphSAGE, TAGCN, Mutation_GAT, Mutation_GCN, Mutation_GraphSAGE, Mutation_TAGCN
from tqdm import tqdm

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_path(path_dir_compile):
    model_path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.pt'):
                    model_path_list.append(file_absolute_path)
    return model_path_list


def load_mutation_model(model_name, path_model, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gat':
        model = Mutation_GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gcn':
        model = Mutation_GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = Mutation_GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = Mutation_TAGCN(num_node_features, hidden_channel, num_classes, dic)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    return model


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

def load_model(model_name, num_node_features, hidden_channel, num_classes,):
    if model_name == 'gat':
        model = GAT(num_node_features, hidden_channel, num_classes)
    elif model_name == 'gcn':
        model = GCN(num_node_features, hidden_channel, num_classes)
    elif model_name == 'graphsage':
        model = GraphSAGE(num_node_features, hidden_channel, num_classes)
    elif model_name == 'tagcn':
        model = TAGCN(num_node_features, hidden_channel, num_classes)
    return model

def select_model(model_name, hidden_channel, num_node_features, num_classes, dic):
    if model_name == 'gat':
        model = Mutation_GAT(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'gcn':
        model = Mutation_GCN(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'graphsage':
        model = Mutation_GraphSAGE(num_node_features, hidden_channel, num_classes, dic)
    elif model_name == 'tagcn':
        model = Mutation_TAGCN(num_node_features, hidden_channel, num_classes, dic)
    return model

def edge_index_to_adj(edge_index_np):
    n_node = max(edge_index_np[0])+1
    m = np.full((n_node, n_node), 0)
    i_j_list = []
    for idx in range(len(edge_index_np[0])):
        i = edge_index_np[0][idx]
        j = edge_index_np[1][idx]
        m[i][j] = 1
        m[j][i] = 1

    return m


def adj_to_edge_index(adj):
    edge_index = np.array(adj.nonzero().t().contiguous().detach().tolist())
    return edge_index


def get_idx_miss_class(target_pre, test_y):
    idx_miss_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != test_y[i]:
            idx_miss_list.append(i)
    # idx_miss_list.append(i)
    return idx_miss_list


def get_n_kill_model(target_pre, mutation_pre_list):
    n_kill_model = []
    for i in range(len(target_pre)):
        n = 0
        for j in range(len(mutation_pre_list)):
            if mutation_pre_list[j][i] != target_pre[i]:
                n += 1
        n_kill_model.append(n)
    return n_kill_model


def get_res_ratio_list(idx_miss_list, select_idx_list, select_ratio_list):
    res_ratio_list = []
    for i in select_ratio_list:
        n = round(len(select_idx_list) * i)
        tmp_select_idx_list = select_idx_list[: n]
        n_hit = len(np.intersect1d(idx_miss_list, tmp_select_idx_list, assume_unique=False, return_indices=False))   # 求集合交集
        ratio = round(n_hit / len(idx_miss_list), 4)   #保留 4 位小数
        res_ratio_list.append(ratio)
    return res_ratio_list


# def load_data(path_x_np, path_edge_index, path_y):
#     x = pickle.load(open(path_x_np, 'rb'))
#     edge_index = pickle.load(open(path_edge_index, 'rb'))
#     y = pickle.load(open(path_y, 'rb'))

#     num_node_features = len(x[0])
#     num_classes = len(set(y))
#     idx_np = np.array(list(range(len(x))))
    
#     train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=17)

#     x = torch.from_numpy(x)
#     edge_index = torch.from_numpy(edge_index)
#     y = torch.from_numpy(y)

#     return num_node_features, num_classes, x, edge_index, y, test_y, train_idx, test_idx


def get_mutation_model_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, model_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    
    #  这两个维度分别为(nodes_num, ) (nodes_num, mutants_num)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([model(x, edge_index).argmax(dim=1).numpy() for model in tqdm(model_list)]).T
    
    # kill mutants标记    kill 记为 1
    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    # 正确、错误节点  二分类标签
    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    #  这两个维度分别为(nodes_num, mutants_num) (nodes_num, ) 
    return feature_np, label_np

def get_mutation_model_edge_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_edge_index_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([target_model(x, torch.from_numpy(tmp_edge_index)).argmax(dim=1).numpy() for tmp_edge_index in tqdm(mutation_edge_index_list)]).T

    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np

def get_mutation_model_attribute_features(num_node_features, target_hidden_channel, num_classes, target_model_path, x, y, edge_index, mutation_attribute_index_list, model_name):
    target_model = load_target_model(model_name, num_node_features, target_hidden_channel, num_classes, target_model_path)
    target_pre = target_model(x, edge_index).argmax(dim=1).numpy()
    mutation_pre_idx_np = np.array([target_model(torch.from_numpy(tmp_x.astype(np.float32)), edge_index).argmax(dim=1).numpy() for tmp_x in mutation_attribute_index_list]).T

    feature_list = []
    for i in range(len(target_pre)):
        tmp_list = []
        for j in range(len(mutation_pre_idx_np[i])):
            if mutation_pre_idx_np[i][j] != target_pre[i]:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        feature_list.append(tmp_list)
    feature_np = np.array(feature_list)

    label_list = []
    for i in range(len(target_pre)):
        if target_pre[i] != y[i]:
            label_list.append(1)
        else:
            label_list.append(0)
    label_np = np.array(label_list)

    return feature_np, label_np

def get_entropy(x):
    # 注意, x 没有被log 返回值
    entropy_score = -np.sum(x * np.log(x), axis=1)  
    entropy_score = np.nan_to_num(entropy_score)
    
    return entropy_score
    
def get_DeepGini(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    return gini_score


#       错误的节点id       排序后的无标节点
def apfd(error_idx_list, pri_idx_list):
    error_idx_list = list(error_idx_list)
    pri_idx_list = list(pri_idx_list)
    n = len(pri_idx_list)
    m = len(error_idx_list)
    
    
    # list.index(i)   i元素 所在的位置
    TF_list = [pri_idx_list.index(i) for i in error_idx_list]    
    apfd = 1 - sum(TF_list)*1.0 / (n*m) + 1 / (2*n)
    return apfd

