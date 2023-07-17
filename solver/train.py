import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network.resnet_modeling import ResNet50 as resNet
from network.config import n_classes
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Hackathon_Barclays', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Hackathon_Barclays', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Hackathon', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=25000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=40, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,help='whether use deterministic training')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input size of network input')
parser.add_argument('--seed', type=int,default=1234, help='random seed')
parser.add_argument('--resnet_name', type=str,default='ResNet50', help='select one resnet model')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Hackathon_Barclays': {
            'root_path': '../data/Hackathon_Barclays',
            'list_dir': './lists/lists_Hackathon',
            'num_classes': 3,
        },
    }
    
    args.num_classes = n_classes
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' 
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    net = resNet(args.num_classes).cuda()
    print('networks defined')

    trainer = {'Hackathon_Barclays': trainer_synapse} 
    trainer[dataset_name](args, net, snapshot_path)