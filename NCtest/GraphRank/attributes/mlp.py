import sys
sys.path.append('../..')

import argparse
import numpy as np
import os
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential

from utils import *

class myDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_out):
        super(MLP,self).__init__()
    
        self.L1 = nn.Linear(n_feature,256)
        self.L2 = nn.Linear(256,64)
        self.L3 = nn.Linear(64,n_out)
       
    def forward(self, x):

        x = self.L1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.L2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.L3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        return F.log_softmax(x, dim=1)
    
    
def train(x, y, num_node_features, num_classes, train_idx, test_idx, epochs, batch_size, save_model_name, save_pre_name, save_attr_name, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(num_node_features, num_classes).to(device)
    x = x.to(device)
    y = y.to(device)

    dataset_train = myDataset(x[train_idx], y[train_idx])
    dataset_test = myDataset(x[test_idx], y[test_idx])

    data_load_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_load_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False) 
            
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in tqdm(range(epochs)):
        for batch_x, batch_y in data_load_train:
            optimizer.zero_grad()
            outs = model(batch_x)
            loss = F.nll_loss(outs, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_model_name)

    model.eval()
    pred = model(x).argmax(dim=1)

    correct = (pred[train_idx] == y[train_idx]).sum()
    acc = int(correct) / len(train_idx)
    print('train:', acc)

    correct = (pred[test_idx] == y[test_idx]).sum()
    acc = int(correct) / len(test_idx)
    print('test:', acc)

    pre = model(x)
    pre = pre.detach().cpu().numpy()
    pre = np.exp(pre)
    pickle.dump(pre, open(save_pre_name, 'wb'), protocol=4)

    entropy = get_entropy(pre)
    mlp_attr = np.concatenate((pre, np.expand_dims(entropy, axis=1)), axis=1)
    np.save(save_attr_name, mlp_attr)
    
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
    
    path_x_np = '../../data/{}/x_np.pkl'.format(dataset_name)
    path_y = '../../data/{}/y_np.pkl'.format(dataset_name)
    
    epochs = 50
    batch_size = 32
    save_model_name = './MLP/{}/mlp_{}_{}.pt'.format(seed, dataset_name, model_name)
    save_pre_name = './MLP/{}/pre_{}_{}.pt'.format(seed, dataset_name, model_name)
    save_attr_name = './MLP/{}/mlpAttr_{}_{}.npy'.format(seed, dataset_name, model_name)

    if not os.path.exists('./MLP/{}'.format(seed)):
        os.makedirs('./MLP/{}'.format(seed), exist_ok=True)
            
    x = pickle.load(open(path_x_np, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    
    num_node_features = len(x[0])
    num_classes = len(set(y))

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    
    idx_np = np.array(list(range(len(x))))
    train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=seed)

    train(x, y, num_node_features, num_classes, train_idx, test_idx, epochs, batch_size, save_model_name, save_pre_name, save_attr_name, seed)