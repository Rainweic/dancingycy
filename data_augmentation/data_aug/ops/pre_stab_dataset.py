import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2
from skimage import feature
import random
import torch.utils.data as data
from ops.base_dataset import get_transform_new, get_params_new,get_transform, get_params
import pickle
from ops.stn import STN

class PreStabDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot                
        data_list_path = os.path.join(opt.dataroot, opt.phase + '.txt')
        f = open(data_list_path, 'r')
        self.all_paths = f.readlines()
        self.dataset_size = len(self.all_paths)
        if (not opt.isTrain and opt.serial_batches):
            ff = open(opt.test_delta_path, "r")
            t = int(ff.readlines()[0])
            self.delta = t
        else:
            self.delta = 0
        if opt.use_H1:
            with open(os.path.join(opt.dataroot, opt.phase + '_H1.p'), 'rb') as f:
                self.H1 = pickle.load(f)
        if opt.use_H2:
            with open(os.path.join(opt.dataroot, opt.phase + '_H2.p'), 'rb') as f:
                self.H2 = pickle.load(f)
            
    def get_X(self, data, params, crop_rate = 1, isTest=False, H1 = None, H2 = None):
        dc = (1 - crop_rate) / 2
        t = data.split('|')
        root = t[0]
        base_n = int(t[1]) + self.opt.start_pos_delta
        if self.opt.new_load_strategy and not isTest:
            transform_B = get_transform_new(self.opt, params)      
        else:
            transform_B = get_transform(self.opt, params)      
        ans = []
        B_tensor_list = []
        for i in range(self.opt.num_segments):
            path = os.path.join(root, str(base_n + i * self.opt.frame_delta) + '.jpg')
            B = Image.open(path).convert('RGB')
            B_tensor_ = transform_B(B)
            B_tensor_list.append(B_tensor_)
        for i in range(self.opt.num_segments):
            if (self.opt.use_diff):
                left = torch.zeros_like(B_tensor_list[i])
                right = torch.zeros_like(B_tensor_list[i])
                if (i > 0):
                    left = (B_tensor_list[i] - B_tensor_list[i - 1]) / 2
                if (i < self.opt.num_segments - 1):
                    right = (B_tensor_list[i] - B_tensor_list[i + 1]) / 2
                B_tensor_ = torch.cat([B_tensor_list[i], left, right], 0)
            else:
                B_tensor_ = B_tensor_list[i]
            _, h, w = B_tensor_.shape
            dh = int(h * dc)
            dw = int(w * dc)
            B_tensor = torch.zeros_like(B_tensor_)
            B_tensor[:, dh : h - dh, dw : w - dw] = B_tensor_[:, dh : h - dh, dw : w - dw]
            ans.append(B_tensor.unsqueeze(0))
        out = torch.cat(ans, 0)#[num_segments, c, h, w]
        return out

    def __getitem__(self, index):        
        index = (index + self.delta) % len(self.all_paths)
        H1 = None
        H2 = None
        if self.opt.use_H1:
            H1 = self.H1[index]
        if self.opt.use_H2:
            H2 = self.H2[index]
        ans = self.get_item(self.all_paths[index], H1, H2)
        return ans

    def get_item(self, data, H1 = None, H2 = None):
        paths = data.rstrip('\n').split('&')
        if (self.opt.new_load_strategy):
            params_1 = get_params_new(self.opt)
            params_2 = get_params_new(self.opt)
        else:
            params_1 = get_params(self.opt, [self.opt.inputSize[1],self.opt.inputSize[0]])
            params_2 = get_params(self.opt, [self.opt.inputSize[1],self.opt.inputSize[0]])

        if self.opt.crop_unstab:
            unstab_tensor = self.get_X(paths[1], params_2, self.opt.crop_rate, False, H1, H2)
        else:
            unstab_tensor = self.get_X(paths[1], params_2, 1, False, H1, H2)
        if not self.opt.no_D:
            stab_tensor = self.get_X(paths[0], params_1, self.opt.crop_rate, False, H1, H2)
        else:
            stab_tensor = torch.zeros_like(unstab_tensor)
        '''
        params_1 = get_params(self.opt, [self.opt.inputSize[1],self.opt.inputSize[0]])
        stab_tensor = self.get_X(paths[1], params_1, 1, True)
        '''
        input_dict = {'stab': stab_tensor, 'unstab': unstab_tensor}
        if self.opt.use_H1:
            input_dict['H1'] = H1
        if self.opt.use_H2:
            input_dict['H2'] = H2
        return input_dict
         
    def get_X_2(self, path, params):
        transform_B = get_transform_new(self.opt, params)      
        B = Image.open(path).convert('RGB')
        B_tensor_ = transform_B(B)
        return B_tensor_
   
    def get_item2(self, path):
        params = get_params_new(self.opt)
        unstab_tensor = self.get_X_2(path, params)
        input_dict = {'unstab': unstab_tensor}
        return input_dict

    def __len__(self):
        return len(self.all_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'PreStabDataset'
