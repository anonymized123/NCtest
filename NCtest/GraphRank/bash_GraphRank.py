import os

with open('GraphRank.sh', 'w') as f: 
    seed_list = range(5)
    dataset_list = ['citeseer', 'cora', 'pubmed', 'coauthorcs']
    # dataset_list = ['coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']

    for seed in seed_list:
        for dataset_name in dataset_list:
            for model_name in model_list:
            
                f.write('python GraphRank.py --seed {} --dataset_name {} --model_name {}\n'.format(seed, dataset_name, model_name))