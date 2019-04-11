### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
import time

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    if (not opt.isTrain) and opt.resize_or_crop == 'scale_width_and_crop':
        x = np.maximum(0, new_w - opt.fineSize) / 2
        y = np.maximum(0, new_h - opt.fineSize) / 2
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def rand_range(lr):
    '''
    if random.random() > 0.5:
        return lr[0]
    else:
        return lr[1]
    '''
    return random.random() * (lr[1] - lr[0]) + lr[0]

def get_params_new(opt):
    return {'brightness': rand_range(opt.load_para_brightness),
            'contrast': rand_range(opt.load_para_contrast),
            'saturation': rand_range(opt.load_para_saturation),
            'hue': rand_range(opt.load_para_hue),
            'angle': rand_range(opt.load_para_angle),
            'shear': rand_range(opt.load_para_shear),
            'scale': rand_range(opt.load_para_scale),
            'translate': [rand_range(opt.load_para_translate),rand_range(opt.load_para_translate)],
            'flip': ((not opt.no_flip) and (random.random() > 0.5)),
            'a_brightness': random.random() < opt.data_argument_rate,
            'a_contrast': random.random() < opt.data_argument_rate,
            'a_saturation': random.random() < opt.data_argument_rate,
            'a_hue': random.random() < opt.data_argument_rate,
            'a_affine': random.random() < opt.data_argument_rate,
            'a_flip': random.random() < opt.data_argument_rate}

def get_transform_new(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if opt.use_diff:
        transform_list.append(transforms.Grayscale())

    if not opt.isTrain:
        if not opt.noResize:
            transform_list.append(transforms.Resize(opt.outputSize, method))   
    else:
        if not opt.noResize:
            transform_list.append(transforms.Resize(opt.outputSize, method))   
        transform_list.append(transforms.Lambda(lambda img: __argument(img, opt, params, method)))
        if opt.isTrain and not opt.no_flip and params['a_flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __argument(img, opt, params, method):
    if params['a_hue']:
        img = F.adjust_hue(img, params['hue'])
    if params['a_saturation']:
        img = F.adjust_saturation(img, params['saturation'])
    if params['a_brightness']:
        img = F.adjust_brightness(img, params['brightness'])
    if params['a_contrast']:
        img = F.adjust_contrast(img, params['contrast'])
    if params['a_affine']:
        img = F.affine(img, params['angle'], params['translate'], params['scale'], params['shear'], resample=method)
    return img

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if opt.use_diff:
        transform_list.append(transforms.Grayscale())
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

