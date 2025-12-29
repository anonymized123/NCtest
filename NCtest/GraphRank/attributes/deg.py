import pickle
import numpy as np
from sklearn.preprocessing import normalize



    # entropy = get_entropy(pre)
    # mlp_attr = np.concatenate((pre, np.expand_dims(entropy, axis=1)), axis=1)
    # np.save(save_attr_name, mlp_attr)
    
if __name__ == "__main__":
    dataset_list = ['coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']

    for dataset_name in dataset_list:
        for model_name in model_list:
            path_x_np = '../../data/{}/x_np.pkl'.format(dataset_name)
            path_y = '../../data/{}/y_np.pkl'.format(dataset_name)
            path_edge_index = '../../data/{}/edge_index_np.pkl'.format(dataset_name)

            save_attr_name = './DEG/degAttr_{}_{}.npy'.format(dataset_name, model_name)

            x = pickle.load(open(path_x_np, 'rb'))
            y = pickle.load(open(path_y, 'rb'))
            edge_index = pickle.load(open(path_edge_index, 'rb'))

            deg_attr = np.zeros(shape=(x.shape[0], 1))
            for i in range(edge_index.shape[1]):
                deg_attr[edge_index[0][i]] += 1
            
            deg_attr = deg_attr / deg_attr.max()
            np.save(save_attr_name, deg_attr)
          

      