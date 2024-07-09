import numpy as np
import torch

from utils.prepare_saved_models import prepare_saved_model
from utils.util import make_dir, get_current_time


class ConfigBasic:
    def __init__(self, target):
        self.pc = 'local' #'anonymous', 'local'
        self.dataset = 'DK'
        self.target = target
        self.backbone = 'vit_b16_reduce' #'vit_b16_reduce', 'vgg16v2norm_reduce'
        self.model = 'GOL'  # 'GOL' or regressor
        self.model_opt = 'GOL' # 'GOLvanilla', 'GOL' , ##GOL uses all three losses, GOLvanilla uses only clustering loss
        self.set_dataset()
        self.set_optimizer_parameters()
        self.set_network()
        self.set_training_opts()

    def set_dataset(self):
        if self.dataset == 'DK': # fixed cropped DK chm set
            if self.pc == 'local':
                self.img_base = '/mnt/ssda/DRIFT/Denmark/images/'
                self.label_path = '/mnt/ssda/DRIFT/Denmark/labels.csv'

            else:
                raise NotImplementedError
            self.img_id = 'filename'
            self.croplength = 200
            self.ground_resolution = 40
            if self.backbone == 'vgg16v2norm_reduce':
                self.inputlength = 200
            elif self.backbone == 'vit_b16_reduce':
                self.inputlength = 224
            self.inputchannel = 3
            self.train_ratio = 0.6
            self.val_ratio = 0.2
            self.test_ratio = 0.2
            self.set_random_seed = True

            if self.target == 'height':
                self.statis = 'nonzero_mean_height(m)'
                self.upper_bound = 25
                self.lower_bound = 0
                self.upper_clip = 50

            elif self.target == 'count':
                self.statis = 'tree_count'
                self.upper_bound = 90
                self.lower_bound = 0
                self.upper_clip = 1000

            elif self.target == 'treecover5m':
                self.statis = 'tree_cover_5m(%)'
                self.upper_bound = 1
                self.lower_bound = 0.2
                self.upper_clip = 1.1

        elif self.dataset == 'france':
            if self.pc == 'local':
                self.img_base = '/mnt/ssda/DRIFT/France/images/'
                self.label_path = '/mnt/ssda/DRIFT/France/labels.csv'
            else:
                raise NotImplementedError
            self.img_id = 'filename'
            self.croplength = 250
            self.ground_resolution = 50
            if self.backbone == 'vgg16v2norm_reduce':
                self.inputlength = 250
            elif self.backbone == 'vit_b16_reduce':
                self.inputlength = 224
            self.inputchannel = 3
            self.train_ratio = 0.6
            self.val_ratio = 0.2
            self.test_ratio = 0.2
            self.set_random_seed = True

            if self.target == 'height':
                self.statis = 'nonzero_mean_height(m)'
                self.upper_bound = 14 # 90 percentile
                self.lower_bound = 0
                self.upper_clip = 50

            elif self.target == 'count':
                self.statis = 'tree_count'
                self.upper_bound = 70
                self.lower_bound = 0
                self.upper_clip = 1000

            elif self.target == 'treecover5m':
                self.statis = 'tree_cover_5m(%)'
                self.upper_bound = 1
                self.lower_bound = 0.2
                self.upper_clip = 1.1

        self.center_cp = False
        self.circle_cp = False
        self.logscale = False
        self.offset = 1
        self.augmentation_prob = 0.5
        self.tau = 1  # rank difference threshold
        self.class_interval_scheme = 'numeric'
        self.class_number = 5 # ranks
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.is_filelist = True

    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.momentum = 0.9
        self.weight_decay = 0.0005
        # *** Scheduler
        self.scheduler = 'multistep'
        self.lr_decay_epochs = [50, 100, 150]
        self.lr_decay_rate = 0.75

    def set_network(self):
        self.backbone_end_dim = 512
        self.ckpt = None

    def set_training_opts(self):
        self.n_gpu = torch.cuda.device_count()
        self.freeze_bn = True
        self.learning_rate = 1e-5
        self.print_freq = 50
        self.val_freq = 2
        self.test_freq = 20
        # *** Training
        self.num_workers = 26
        self.epochs = 200
        # *** Save option
        self.wandb = True
        self.project_name = '{}-{}-fixed-classification-v2'.format(self.dataset, self.target)
        self.experiment_name = '{}_{}_{}_more_aug_{}'.format(self.statis, self.model_opt, self.backbone, get_current_time()).replace(' ', '_')
        self.resume_model = False  # finetune initally trained model
        if self.resume_model:
            # find out model path in prepare_saved_model(_anonymous)
            model_name = self.dataset + '-' + self.target + '-' + self.backbone[:3] + '-' + self.model_opt
            if self.pc == 'local':
                self.resume_model_path = prepare_saved_model(model_name)

        if self.model == 'GOL':
            self.batch_size = 32
            self.batch_size_pred = 32  # smaller batch size works better with large num workers!!
            self.metric = 'L2'
            self.init_knn_k = 10  # fix k for search
            self.order_loss_func = 'Focal'  # or 'CE'
            self.focal_loss_gamma = 3
            self.margin = 0.25 # change margin to be a mupltipler instead of an addend
            self.ref_mode = 'flex_reference'
            self.ref_point_num = 5  # same as number of ranks
            if self.model_opt == 'GOL':
                self.drct_wieght = 1 # weight for order loss, make the sum of 3 to be 100
                self.metric_los_wei = 66
                self.center_los_wei = 33
            elif self.model_opt == 'GOLvanilla':
                self.drct_wieght = 0
                self.metric_los_wei = 0
                self.center_los_wei = 1
            self.start_norm = True
            self.stage = 1
            self.predict_scheme = 'weighted-mean'
            self.alpha = 3
            self.alpha_n = 20  # 3
            self.reciprocal = False
            self.reciprocal_k1 = 20
            self.reciprocal_k2 = 10  # 3
            self.reciprocal_l = 0.8
            if self.pc == 'local':
                self.save_folder = f'/home/sizhuo/Desktop/code_repository/domain_adaptive_regression/saved_models/{self.dataset}/{self.experiment_name}/tau{self.tau}_metric{self.metric}_{self.model}_{self.backbone}'
            else:
                raise NotImplementedError
            make_dir(self.save_folder)

        elif self.model == 'regressor':
            # direct regression to target variable
            self.batch_size = 64
            self.batch_size_pred = 64  # smaller batch size works better with large num workers!!
            self.loss_func = 'MAE'
            self.num_linear = 1 # number of linear layers on top of backbone
            self.add_dropout = False

            if self.pc == 'local':
                self.save_folder = f'/home/sizhuo/Desktop/code_repository/domain_adaptive_regression/saved_models/{self.dataset}/{self.experiment_name}/{self.model}_{self.backbone}'
            elif self.pc == 'anonymous':
                self.save_folder = f'/home/sizhuo/Desktop/code_repository/domain_adaptive_regression/saved_models/{self.dataset}/{self.experiment_name}/{self.model}_{self.backbone}'
            make_dir(self.save_folder)

