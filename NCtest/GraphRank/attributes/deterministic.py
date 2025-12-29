import sys
sys.path.append('../..')

import numpy as np
from sklearn.preprocessing import normalize
import argparse
import pickle

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed 
    
    # set_seeds(seed)
    
    dataset_list = ['coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']

    for dataset_name in dataset_list:
        for model_name in model_list:
            path_x_np = '../../data/{}/x_np.pkl'.format(dataset_name)
            path_y = '../../data/{}/y_np.pkl'.format(dataset_name)
            path_edge_index = '../../data/{}/edge_index_np.pkl'.format(dataset_name)
            target_model_path = '../../GNN_train/models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
                        
            save_attr_name = './DETER/{}/deterAttr_{}_{}.npy'.format(seed, dataset_name, model_name)
            
            if not os.path.exists('./DETER/{}'.format(seed)):
                os.makedirs('./DETER/{}'.format(seed), exist_ok=True)
                
            ################# load data
            x = pickle.load(open(path_x_np, 'rb'))
            edge_index = pickle.load(open(path_edge_index, 'rb'))
            y = pickle.load(open(path_y, 'rb'))

            num_node_features = len(x[0])
            num_classes = len(set(y))
            num_node = x.shape[0]
            idx_np = np.array(list(range(len(x))))
            train_idx, test_idx, train_y, test_y = train_test_split(idx_np, y, test_size=0.3, random_state=seed)

            x = torch.from_numpy(x)
            edge_index = torch.from_numpy(edge_index)
            y = torch.from_numpy(y)
            #################
            
            target_model = load_target_model(model_name, num_node_features, 16, num_classes, target_model_path)
            target_probability = np.exp(target_model(x, edge_index).detach().numpy())
            
            entropy_score = get_entropy(target_probability)
            gini_score = get_DeepGini(target_probability)
            gini_score = get_DeepGini(target_probability)
            
            deter_attr = np.concatenate((target_probability, np.expand_dims(entropy_score, axis=1), np.expand_dims(gini_score, axis=1)), axis=1)

            np.save(save_attr_name, deter_attr)
