### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='stab_pre_0', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        '''
        self.parser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        '''
        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--inputSize', type=str, default='720,1280', help='scale images to this size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--outputSize', type=str, default='288,512', help='then crop to this size')
        self.parser.add_argument('--noResize', action='store_true', help='if specified, use diff') 
        self.parser.add_argument('--num_segments', type=int, default=5, help='then crop to this size')
        self.parser.add_argument('--frame_delta', type=int, default=7, help='then crop to this size')
        self.parser.add_argument('--use_diff', action='store_true', help='if specified, use diff') 
        self.parser.add_argument('--limit_board', action='store_true', help='if specified, limit board') 

        '''
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--label_indexs', type=str, default='', help='label index ids, empty means 0..label_nc-1')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        '''
        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./data/pre/') 
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')                
        #for stabnet
        self.parser.add_argument('--grid_hw', type=str, default='2,2', help='scale images to this size')
        self.parser.add_argument('--img_feature_dim', default=256, type=int, help='# threads for loading data')                
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--crop_rate', type=float, default=0.8, help='then crop to this size')
        self.parser.add_argument('--crop_unstab', action='store_true', help='if specified, do not flip the images for data argumentation') 
        #new load strategy
        self.parser.add_argument('--new_load_strategy', action='store_true', help='if specified, limit board') 
        self.parser.add_argument('--data_argument_rate', type=float, default=0.2, help='then crop to this size')
        #inputSize
        #outputSize
        self.parser.add_argument('--load_para_brightness', type=str, default='0.7,1.3', help='0 - 2')
        self.parser.add_argument('--load_para_contrast', type=str, default='0.7,1.3', help='0 - 2+')
        self.parser.add_argument('--load_para_saturation', type=str, default='0.5,1.7', help='0 - 2+')
        self.parser.add_argument('--load_para_hue', type=str, default='-0.05,0.05', help='-0.5 - 0.5')
        self.parser.add_argument('--load_para_angle', type=str, default='-10,10', help='-180 - 180')
        self.parser.add_argument('--load_para_translate', type=str, default='-0.1,0.1', help='move')
        self.parser.add_argument('--load_para_shear', type=str, default='-5,5', help='-180 - 180?')
        self.parser.add_argument('--load_para_scale', type=str, default='1.3,1.5', help='-180 - 180?')

        self.parser.add_argument('--H_version', type=int, default=53, help='-180 - 180?')
        self.parser.add_argument('--H_base_model', type=str, default='resnet34', help='-180 - 180?')
        self.parser.add_argument('--pre_version', type=int, default=56, help='-180 - 180?')
        #train
        self.parser.add_argument('--acceleration_delta_r', type=int, default=6, help='-180 - 180?')
        self.parser.add_argument('--test_TSM', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--test_TSM_no_freeze_f1', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--freeze_f1', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--freeze_f2', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--freeze_flast', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--freeze_f16', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--freeze_first_piece', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--no_D', action='store_true', help='continue training: load the latest model')

        self.parser.add_argument('--piece_size', type=int, default=6, help='-180 - 180?')
        self.parser.add_argument('--lr_list', type=str, default='', help='-180 - 180?')
        self.parser.add_argument('--infer_only', action='store_true', help='continue training: load the latest model')

        self.parser.add_argument('--start_pos_delta', type=int, default=0, help='-180 - 180?')
        self.parser.add_argument('--use_H1', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--H1_list', type=str, default='', help='-180 - 180?')
        self.parser.add_argument('--use_H2', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--H2_list', type=str, default='', help='-180 - 180?')
        self.parser.add_argument('--freeze_list', type=str, default='', help='-180 - 180?')
        self.parser.add_argument('--TRN_num_bottleneck', type=int, default=512, help='-180 - 180?')

        '''
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')        
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')        

        self.parser.add_argument('--no_canny_edge', action='store_true', help='no canny edge for background')        
        self.parser.add_argument('--no_dist_map', action='store_true', help='no dist map for background')        
        self.parser.add_argument('--do_pose_dist_map', action='store_true', help='do pose dist map for background')        

        self.parser.add_argument('--remove_face_labels', action='store_true', help='remove face labels to better adapt to different face shapes')
        self.parser.add_argument('--random_drop_prob', type=float, default=0.2, help='the probability to randomly drop each pose segment during training')

        self.parser.add_argument('--densepose_only', action='store_true', help='use only densepose as input')
        self.parser.add_argument('--openpose_only', action='store_true', help='use only openpose as input') 
        self.parser.add_argument('--add_mask', action='store_true', help='mask of background') 

        self.parser.add_argument('--vgg_weights', type=str, default='1,1,1,1,1', help='vgg weights of ans&guidence loss')
        self.parser.add_argument('--gram_weights', type=str, default='1,1,1,1,1', help='gram weights of ans&guidence loss')
        self.parser.add_argument('--guide_vgg_mul', type=float, default=0.0, help='')
        self.parser.add_argument('--guide_gram_mul', type=float, default=0.0, help='')
        self.parser.add_argument('--use_self_loss', action='store_true', help='mask of background') 
        self.parser.add_argument('--self_vgg_weights', type=str, default='1,1,1,1,1', help='vgg weights of ans&guidence loss')
        self.parser.add_argument('--self_gram_weights', type=str, default='1,1,1,1,1', help='gram weights of ans&guidence loss')
        self.parser.add_argument('--self_vgg_mul', type=float, default=0.0, help='')
        self.parser.add_argument('--self_gram_mul', type=float, default=0.0, help='')

        self.parser.add_argument('--style_stage_mul', type=str, default='0:1', help='')
        self.parser.add_argument('--real_stage_mul', type=str, default='0:1', help='')
        self.parser.add_argument('--train_val_list', type=str, default='0,1,2,3,4,100000,200000,300000,400000,500000', help='')

        self.parser.add_argument('--no_D_label', action='store_true', help='remove label channel of input of D')        
        self.parser.add_argument('--no_G_label', action='store_true', help='remove label channel of input of G')        
        self.parser.add_argument('--use_new_label', action='store_true', help='use generated seg map')        
        '''
        self.initialized = True

    def get_list_int(self, s):
        str_indexs = s.split(',')
        ans = []
        if (str_indexs[0] != ''):
            for str_index in str_indexs:
                ans.append(int(str_index))
        return ans

    def get_list_float(self, s):
        str_indexs = s.split(',')
        ans = []
        if (str_indexs[0] != ''):
            for str_index in str_indexs:
                ans.append(float(str_index))
        return ans

    def get_list_dict(self, s):
        str_indexs = s.split(',')
        ans = []
        if (str_indexs[0] != ''):
            for str_index in str_indexs:
                temp = str_index.split(':')
                ans.append([int(temp[0]), float(temp[1])])
        return ans

    def get_list_dict2(self, s):
        str_indexs = s.split(',')
        ans = []
        if (str_indexs[0] != ''):
            for str_index in str_indexs:
                temp = str_index.split(':')
                ans.append([int(temp[0]), int(temp[1])])
        return ans

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.gpu_ids = self.get_list_int(self.opt.gpu_ids)
        self.opt.grid_hw = self.get_list_int(self.opt.grid_hw)
        self.opt.inputSize = self.get_list_int(self.opt.inputSize)
        self.opt.outputSize = self.get_list_int(self.opt.outputSize)

        self.opt.load_para_brightness = self.get_list_float(self.opt.load_para_brightness) 
        self.opt.load_para_contrast = self.get_list_float(self.opt.load_para_contrast) 
        self.opt.load_para_saturation = self.get_list_float(self.opt.load_para_saturation) 
        self.opt.load_para_hue = self.get_list_float(self.opt.load_para_hue) 
        self.opt.load_para_angle = self.get_list_float(self.opt.load_para_angle) 
        self.opt.load_para_translate = self.get_list_float(self.opt.load_para_translate) 
        self.opt.load_para_shear = self.get_list_float(self.opt.load_para_shear) 
        self.opt.load_para_scale = self.get_list_float(self.opt.load_para_scale) 

        self.opt.H_regressor_path = './checkpoints/stab_pre_' + str(self.opt.H_version) + '/'
        self.opt.pre_G_path = './checkpoints/stab_pre_' + str(self.opt.pre_version) + '/'

        self.opt.H1_list = self.get_list_dict2(self.opt.H1_list)
        self.opt.H2_list = self.get_list_dict2(self.opt.H2_list)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        # set lr list
        if self.opt.lr_list == '':
            self.opt.lr_list = [[0, self.opt.lr]]
        else:
            self.opt.lr_list = self.get_list_dict(self.opt.lr_list)

        # set freeze list
        self.opt.freeze_list = self.get_list_int(self.opt.freeze_list)
        if (self.opt.freeze_f1):
            self.opt.freeze_list.append(0)
        if (self.opt.freeze_f2):
            self.opt.freeze_list.append(1)
        if (self.opt.freeze_f16):
            self.opt.freeze_list.append(15)
        if (self.opt.freeze_flast):
            self.opt.freeze_list.append(self.opt.num_segments - 1)
        if (self.opt.freeze_first_piece):
            for i in range(self.opt.piece_size):
                self.opt.freeze_list.append(i)
        self.opt.freeze_list = list(set(self.opt.freeze_list))
        self.opt.freeze_list.sort()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        '''
        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        '''
        return self.opt
