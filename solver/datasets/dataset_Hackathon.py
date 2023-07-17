import os
import random
import h5py
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from medpy import metric
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from utils import encoder,decoder,padded_img

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        #image = padded_img(image)        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label}
       
        return sample
    
class Hackathon_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.image_list = open(os.path.join(list_dir, 'training_image'+'.txt')).readlines() 
        self.label_list = open(os.path.join(list_dir, 'training_label'+'.txt')).readlines()
        
        self.test_image_list = open(os.path.join(list_dir, 'testing_image'+'.txt')).readlines()
        self.test_label_list = open(os.path.join(list_dir, 'testing_label'+'.txt')).readlines()

        
    def __len__(self):
        return len(self.image_list) if self.split == 'train' else len(self.test_image_list)
    
    def __getitem__(self, idx):
        if self.split == "train":
            image_name = self.image_list[idx].strip('\n')
            label_name = self.label_list[idx].strip('\n')
            
            image_path = os.path.join(self.data_dir,'test', image_name)
            
            image_array = Image.open(image_path)
            image = np.array(image_array)
           
            image = image/255.0
            
            label = encoder(label_name)
            
            sample = {'image': image, 'label': label}
            sample['case_name'] = self.image_list[idx].strip('\n')

        else:
                       
            image_name = self.test_image_list[idx].strip('\n')
            label_name = self.test_label_list[idx].strip('\n')
            
            image_path = os.path.join(self.data_dir,'Images', image_name+'.png')
            
           
            image_array = Image.open(image_path)
            image = np.array(image_array)
            
            image = image/255.0
            
            label = encoder(label_name)
            label = torch.from_numpy(label)
            sample = {'image': image, 'label': label}
            sample['case_name'] = self.test_image_list[idx].strip('\n')

        if self.transform:
            sample = self.transform(sample)

        
        return sample