import sys
sys.path.append('../..')

import argparse
import numpy as np
from sklearn.preprocessing import normalize

from utils import *

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
    path_edge_index = '../../data/{}/edge_index_np.pkl'.format(dataset_name)
    target_model_path = '../../GNN_train/models_save/{}/{}_{}.pt'.format(seed, dataset_name, model_name)
    
    save_attr_name = './PRO/{}/probaAttr_{}_{}.npy'.format(seed, dataset_name, model_name)

    if not os.path.exists('./PRO/{}'.format(seed)):
        os.makedirs('./PRO/{}'.format(seed), exist_ok=True)
        
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
    target_model.train()
    
    n = 10
    probability_n = np.zeros(shape=(num_node, num_classes))
    for i in range(n):
        prob_out = target_model(x, edge_index).detach().numpy()
        # print(prob_out)
        probability_n += np.exp(prob_out)
    probability_n /= 10
    
    uncertainty_score = -np.sum(probability_n * np.log(probability_n), axis=1)  
    uncertainty_score = np.expand_dims(uncertainty_score, axis=1)
    np.save(save_attr_name, uncertainty_score)



      