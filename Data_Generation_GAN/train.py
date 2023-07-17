import argparse
import torch as t
from trainer import trainer_synapse
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.discriminator import Discriminator_Network as disc
from networks.generator import Generator_Network as gan


parser = argparse.ArgumentParser(description='PyTorch Implementation of a GAN')
parser.add_argument('--dataset', type=str,
                default='Hackathon_Barclays', help='experiment_name')
parser.add_argument('--root_path', type=str,
                default='../data/Hackathon_Barclays', help='root dir for data')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train for (default: 300)')
parser.add_argument('--learning_rate_gan', type=float, default=1e-4,
                    help='learning rate for Generator (default: 1e-4)')
parser.add_argument('--learning_rate_disc', type=float, default=1e-4,
                    help='learning rate for Discriminator (default: 1e-4)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='number of examples in a batch (default: 32)')
parser.add_argument('--device', type=int, default=t.device("cuda:0" if t.cuda.is_available() else "cpu"),
                    help='device to train on (default: cuda:0 if cuda is available otherwise cpu)')

parser.add_argument('--latent-size', type=int, default=64,
                    help='size of latent space vectors (default: 64)')
parser.add_argument('--g-hidden-size', type=int, default=256,
                    help='number of hidden units per layer in G (default: 256)')
parser.add_argument('--d-hidden-size', type=int, default=256,
                    help='number of hidden units per layer in D (default: 256)')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

args = parser.parse_args()
    
if __name__ == '__main__':
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
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../GAN_Model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' 
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    gan_model = gan(200*50, args.latent_size, args.g_hidden_size).to(args.device)
    disc_model = disc(200*50, args.d_hidden_size).to(args.device)
    print('networks defined')
    trainer = {'Hackathon_Barclays': trainer_synapse} 
    trainer[dataset_name](args, gan_model, disc_model, snapshot_path)