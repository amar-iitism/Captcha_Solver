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
from torchvision import transforms
from datasets.dataset_Hackathon import Hackathon_dataset, RandomGenerator

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


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir, transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    dataset_length = len(db_test)

    print("Length of db_test dataset:", dataset_length)
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=8)
    count_matching =0
    count_total = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()            
        outputs = model(image_batch)
        outputs = F.softmax(x, dim=1)
        outputs = torch.argmax(probabilities, dim=1)
        output_batch =[]
        
        for i in range (0,outputs[0].
            extracted_decoder = decoder(tensor[i, :])
            output_batch.append(extracted_decoder)
        for item1, item2 in zip(label_batch, output_batch):
            count_total +=1
            if item1 == item2:
            count_matching += 1
                        
    accuracy = ((count_matching)/(count_total))*100
    logging.info('Testing performance in best val model: accuracy: %f' % (accuracy))
    return "Testing Finished!"




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
    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    
    inference(args, net, snapshot_path)
    
    