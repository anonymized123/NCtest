import os

with open('train_improve.sh', 'w') as f: 
    seed_list = [0,1,2,3,4]
    dataset_list = ['citeseer', 'cora', 'pubmed', 'coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']
    # method_list = ['random', 'entropy', 'margin', 'deepGini', 'dropout', 'xgb', 'stacking', 'gr_xgb']
    method_list = ['qkv']
    for seed in seed_list:
        for method in method_list:
            for dataset_name in dataset_list:
                for model_name in model_list:
                        f.write('python train_improve.py --seed {} --dataset_name {} --model_name {} --method {}\n'.format(seed, dataset_name, model_name, method))
                