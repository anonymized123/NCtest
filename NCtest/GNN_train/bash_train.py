import os

with open('train.sh', 'w') as f: 
    seed_list = [0,1,2,3,4,5,6,7,8,9]
    dataset_list = ['citeseer', 'cora', 'pubmed', 'coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']

    for seed in seed_list:
        for dataset_name in dataset_list:
            for model_name in model_list:
                f.write('python train.py --seed {} --dataset_name {} --model_name {}\n'.format(seed, dataset_name, model_name))
                