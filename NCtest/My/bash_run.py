import os

path = 'attn'


with open('run.sh', 'w') as f: 
    f.write(f'mkdir result_{path}\n')
    
    seed_list = [0,1,2,3,4]
    # seed_list = [0,1,2]
    dataset_list = ['citeseer', 'cora', 'pubmed', 'coauthorcs']
    # dataset_list = ['coauthorcs']
    model_list = ['gat', 'gcn', 'graphsage', 'tagcn']
    for seed in seed_list:
        f.write('rm -r ./result_{}/apfd_{}.csv\n'.format(path, seed))
                
    for seed in seed_list:
        for dataset_name in dataset_list:
            for model_name in model_list:
            
                f.write('python main_{}.py --seed {} --dataset_name {} --model_name {}\n'.format(path, seed, dataset_name, model_name))
                
    f.write('python get_mean.py --path {}'.format(path))
    