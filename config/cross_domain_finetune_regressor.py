import os.path

import ipdb
import numpy as np
import torch

from utils.util import make_dir, get_current_time
from utils.prepare_saved_models import prepare_saved_model


class Config:
    def __init__(self, source_dataset, infer_dataset, target, backbone, few_shot_num):
        self.pc = 'local'
        self.source_dataset = self.dataset = source_dataset
        self.infer_dataset = infer_dataset
        self.target = target
        self.model = 'regressor'
        self.model_opt = 'regressor' # GOL or GOLvanilla
        self.backbone = backbone #'vit_b16_reduce' #'vgg16v2norm_reduce'
        self.backbone_short = self.backbone[:3]
        self.few_shot_num = few_shot_num # shots per class, n_ranks ways
        if self.few_shot_num <= 10:
            self.few_shot_valid_num = 1
        elif self.few_shot_num == 20:
            self.few_shot_valid_num = 3
        elif self.few_shot_num == 30:
            self.few_shot_valid_num = 5
        self.logscale = False # whether regress directly to the real value
        self.set_dataset()
        self.set_network()
        self.set_finetune_opts()
        self.set_optimizer_parameters()

    def set_dataset(self):
        # source domain dataset
        if self.source_dataset == 'DK':
            # config of source data (training in-domain data)
            if self.pc == 'local':
                self.label_path = '/mnt/ssda/DRIFT/Denmark/labels.csv'  # train and val
            elif self.pc == 'anonymous':
                raise NotImplementedError
            self.img_id = 'filename'
            self.croplength = 200
            self.base_resolution = 0.2  # resolution of the training data
            self.ground_resolution = 40
            if self.backbone == 'vgg16v2norm_reduce':
                self.inputlength = 200
            elif self.backbone == 'vit_b16_reduce':
                self.inputlength = 224
            self.train_ratio = 0.8
            self.set_random_seed = True

        elif self.source_dataset == 'france':
            if self.pc == 'local':
                self.label_path = '/mnt/ssda/DRIFT/France/labels.csv'
            elif self.pc == 'anonymous':
                raise NotImplementedError
            self.img_id = 'filename'
            self.croplength = 250
            self.base_resolution = 0.2  # resolution of the training data
            self.ground_resolution = 50
            if self.backbone == 'vgg16v2norm_reduce':
                self.inputlength = 250
            elif self.backbone == 'vit_b16_reduce':
                self.inputlength = 224
            self.train_ratio = 0.8
            self.set_random_seed = True

        else:
            raise NotImplementedError


        # infer dataset
        if self.infer_dataset == 'SP':
            if self.pc == 'local':
                self.infer_label_path = '/mnt/ssda/DRIFT/Spain/labels.csv'
                self.img_dir = '/mnt/ssda/DRIFT/Spain/images/'
            elif self.pc == 'anonymous':
                raise NotImplementedError

            self.img_filter_check = False
            self.wrongFileRemove = False  # no wrong file list, file not checked for fixed crops
            self.wrongFilePath = None
            self.infer_resolution = 0.25  # resolution of the validation data (cross area)
            self.infer_ground_resolution = 50  # 50m NFI plot

        elif self.infer_dataset == 'slovenia':
            if self.pc == 'local':
                self.infer_label_path = '/mnt/ssda/DRIFT/Slovenia/labels.csv'
                self.img_dir = '/mnt/ssda/DRIFT/Slovenia/images/'
            elif self.pc == 'anonymous':
                raise NotImplementedError
            self.img_filter_check = False  # check if image file is complete
            self.wrongFileRemove = False  # no wrong file list, file not checked for fixed crops
            self.wrongFilePath = None
            self.infer_resolution = 0.5  # resolution of the validation data (cross area)
            self.infer_ground_resolution = 50  # 50m

        elif self.infer_dataset == 'slovakia':
            if self.pc == 'local':
                self.infer_label_path = '/mnt/ssda/DRIFT/Slovakia/labels.csv'
                self.img_dir = '/mnt/ssda/DRIFT/Slovakia/images/'
            elif self.pc == 'anonymous':
                raise NotImplementedError
            self.img_filter_check = False  # check if image file is complete
            self.wrongFileRemove = False  # no wrong file list, file not checked for fixed crops
            self.wrongFilePath = None
            self.infer_resolution = 0.2  # resolution of the validation data (cross area)
            self.infer_ground_resolution = 50  # 50m

        else:
            raise NotImplementedError

        # target variable config
        if self.target == 'height':
            self.statis = 'nonzero_mean_height(m)'
            if self.source_dataset == 'DK':
                self.upper_bound = 25 # only concerns in-domain test
            elif self.source_dataset == 'france':
                self.upper_bound = 14
            self.lower_bound = 0
            self.upper_clip = 50
            # infer
            self.attribute = 'nonzero_mean_height(m)'
            self.infer_upper_clip = 50


        elif self.target == 'count':

            self.statis = 'tree_count'
            if self.source_dataset == 'DK':
                self.upper_bound = 90
            elif self.source_dataset == 'france':
                self.upper_bound = 70
            self.lower_bound = 0
            self.upper_clip = 1000
            self.attribute = 'tree_count'
            self.infer_upper_clip = 1000


        elif self.target == 'treecover5m':
            self.statis = 'tree_cover_5m'
            self.rescale = 1 # set to 100 to rescale from 0-1 to 0-100
            self.upper_clip = 1.1 * self.rescale
            self.upper_bound = 1 * self.rescale
            self.lower_bound = 0.2 * self.rescale
            self.attribute = 'tree_cover_5m'
            self.infer_upper_clip = 1.1 * self.rescale

        else:
            raise NotImplementedError


        self.fileid = 'fileID'
        self.imagepath = 'filename'
        self.infer_class_number = 5
        self.infer_use_ref_rank = False  # use reference rank for infer data
        self.infer_ranks = None  # ranks for split infer labels
        self.infer_croplength = int(
            self.infer_ground_resolution / self.infer_resolution)  # crop length of the validation data (cross area)
        if self.backbone == 'vgg16v2norm_reduce':
            self.infer_inputlength = self.infer_croplength  # validation data (cross area)
        elif self.backbone == 'vit_b16_reduce':
            self.infer_inputlength = 224 # vit requires 224x224 input

        self.infer_center_cp = False
        self.infer_circle_cp = False
        self.is_filelist = True
        self.inputchannel = 3
        self.infer_upper_bound = 99
        self.infer_lower_bound = 1
        self.center_cp = False #
        self.circle_cp = False # only applies to infer data, circular NIF plots
        self.normalize_target = False # normalize target to [0, 1] based on max and min
        self.offset = 1
        self.tau = 1  # rank difference threshold
        self.class_interval_scheme = 'numeric'
        self.class_number = 5
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def set_network(self):
        self.ckpt = None
        self.backbone_end_dim = 512 # not used
        self.num_linear = 1

        model_name = self.source_dataset + '-' + self.target + '-' + self.backbone_short + '-' + self.model_opt
        if self.pc == 'local':
            self.model_path = prepare_saved_model(model_name)
        else:
            raise NotImplementedError
        print('model name: ', model_name)
        print('model path: ', self.model_path)

        self.same_domain_inds_path = os.path.join(os.path.dirname(self.model_path), 'train-val-tst-idx_dict.json')



    def set_optimizer_parameters(self):
        # *** Optimizer
        self.optim = 'adam'
        self.SAM = True # sharpness-aware minimization
        self.rho = 0.05 # for sam, same as in the fine-tuning paper
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.lr_list = [1e-5, 1e-6, 1e-7, 1e-8]
        self.learning_rate = None
        self.loss_fn = 'mae' # mse or l1



    def set_finetune_opts(self):
        self.repeat = 3
        self.n_gpu = torch.cuda.device_count()
        self.batch_size = 32
        self.batch_size_pred = 32 # smaller batch size works better with large num workers!!
        self.augmentation_prob = 0.5
        self.num_workers = 0
        self.sample_equal_infer = False
        self.freeze_bn = True
        self.val_freq = 1
        self.test_freq = 1
        self.iterations = 100
        self.print_freq = 20
        self.wandb = True
        self.project_name = 'end2end_{}shot_regressor_finetune-source-{}-inferdata-{}-target-{}-backbone-{}'.format(self.few_shot_num, self.source_dataset, self.infer_dataset, self.target, self.backbone_short)
        self.experiment_name = 'run_{}'.format(get_current_time()).replace(' ', '_')
        self.resume_model = False
        self.save_folder = f'/home/sizhuo/Desktop/code_repository/GOL/saved_models/{self.source_dataset}/{self.experiment_name}'
        make_dir(self.save_folder)
