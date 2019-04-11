### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--check_GD_freq', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--check_GD_threshold', type=float, default=0.6, help='frequency of showing training results on console')
        self.parser.add_argument('--iter_theta_only', type=int, default=500, help='frequency of showing training results on console')
        self.parser.add_argument('--w_loss_G_theta', type=float, default=0.3, help='frequency of showing training results on console')
        self.parser.add_argument('--w_loss_G_pix', type=float, default=0, help='frequency of showing training results on console')
        self.parser.add_argument('--w_loss_G_acceleration', type=float, default=0, help='frequency of showing training results on console')
        self.parser.add_argument('--w_loss_G_L2_acceleration', type=float, default=0, help='frequency of showing training results on console')
        self.parser.add_argument('--w_loss_G_consistency', type=float, default=0, help='frequency of showing training results on console')
        self.parser.add_argument('--save_last_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_freq', type=int, default=50000, help='frequency of saving the latest results')
        self.parser.add_argument('--resume', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--no_alter_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--no_theta_iter', type=int, default=100000000, help='frequency of saving the latest results')
        self.parser.add_argument('--set_iter', type=int, default=-1, help='frequency of saving the latest results')
        self.parser.add_argument('--use_point_acceleration', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--test_mode', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--max_loss_num', type=int, default=1, help='frequency of saving the latest results')

        #for testing
        self.parser.add_argument('--test_freq', type=int, default=50000, help='frequency of saving the latest results')
        self.parser.add_argument('--test_num', type=int, default=50, help='frequency of saving the latest results')
        '''
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_iter_freq', type=int, default=150000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--val_freq', type=int, default=500, help='frequency of showing training results on console')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--use_iter_decay', action='store_true', help='user iter decay')
        self.parser.add_argument('--niter_iter', type=int, default=200000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay_iter', type=int, default=200000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--use_style_iter', type=int, default=-1, help='# of iter to use style discriminator')


        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--nsdf', type=int, default=128, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--SD_mul', type=float, default=10, help='style discriminator mul')
        self.parser.add_argument('--GAN_Feat_mul', type=float, default=1, help='GAN Feat mul')
        self.parser.add_argument('--G_confidence_mul', type=float, default=1, help='G confidence mul')
        self.parser.add_argument('--FG_GAN_mul', type=float, default=10, help='G confidence mul')
        self.parser.add_argument('--val_n_everytime', type=int, default=10, help='val num everytime')

        self.parser.add_argument('--no_SD_false_pair', action='store_true', help='')        

        self.parser.add_argument('--use_stage_lr', action='store_true', help='')        
        self.parser.add_argument('--stage_lr_decay_iter', type=int, default=50000, help='')        
        self.parser.add_argument('--stage_lr_decay_rate', type=float, default=0.3, help='')        
        '''
        self.isTrain = True
