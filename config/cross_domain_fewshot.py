import os.path
import torch
from utils.prepare_saved_models import prepare_saved_model


class Config:
    def __init__(self, source_dataset, infer_dataset, target, backbone, few_shot_num):
        self.source_dataset = source_dataset #'DK', 'france'
        self.infer_dataset = infer_dataset #'SP', 'slovakia', 'slovenia'
        self.target = target #'height', 'count' or 'treecover5m'
        self.model = 'GOL' #'GOL' or 'regressor'
        self.model_opt = 'GOL' # GOL or GOLvanilla (only knn loss) or regressor
        self.backbone = backbone #'vgg16v2norm_reduce' #'vit_b16_reduce' #'vgg16v2norm_reduce'
        self.backbone_short = self.backbone[:3]
        self.few_shot = True
        self.few_shot_num = few_shot_num # shots per class, n_ranks ways
        self.init_knn_k = self.few_shot_num
        self.diffuse_k_value = int(self.few_shot_num * 2)
        if self.model == 'regressor':
            self.logscale = False
        else:
            self.logscale = True
        self.visualize_emb = False # use 3-d model output for visualization
        self.set_dataset()
        self.set_network()
        self.set_valid_opts()

    def set_dataset(self):
        # source domain dataset
        if self.source_dataset == 'DK':
            # config of source data (training in-domain data)
            self.label_path = '/mnt/ssda/DRIFT/Denmark/labels.csv'  # train and val
            self.img_base = '/mnt/ssda/DRIFT/Denmark/images/'
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

            self.img_base = '/mnt/ssda/DRIFT/France/images/'
            self.label_path = '/mnt/ssda/DRIFT/France/labels.csv'
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

            self.infer_label_path = '/mnt/ssda/DRIFT/Spain/labels.csv'
            self.img_dir = '/mnt/ssda/DRIFT/Spain/images/'
            self.img_filter_check = False
            self.wrongFileRemove = False  # no wrong file list, file not checked for fixed crops
            self.wrongFilePath = None
            self.infer_resolution = 0.25  # resolution of the validation data (cross area)
            self.infer_ground_resolution = 50  # 50m NFI plot

        elif self.infer_dataset == 'slovenia':

            self.infer_label_path = '/mnt/ssda/DRIFT/Slovenia/labels.csv'
            self.img_dir = '/mnt/ssda/DRIFT/Slovenia/images/'
            self.img_filter_check = False  # check if image file is complete
            self.wrongFileRemove = False  # no wrong file list, file not checked for fixed crops
            self.wrongFilePath = None
            self.infer_resolution = 0.5  # resolution of the validation data (cross area)
            self.infer_ground_resolution = 50  # 50m


        elif self.infer_dataset == 'slovakia':

            self.infer_label_path = '/mnt/ssda/DRIFT/Slovakia/labels.csv'
            self.img_dir = '/mnt/ssda/DRIFT/Slovakia/images/'
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
        if self.model == 'regressor':
            self.num_linear = 1
        if self.visualize_emb:
            self.backbone_end_dim = 3
            # take model that outputs 3 dim embedding
            model_name = self.source_dataset + '-' + self.target + '-' + self.backbone_short + '-' + self.model_opt + '-reduced'
        else:
            self.backbone_end_dim = 512  # 512
            model_name = self.source_dataset + '-' + self.target + '-' + self.backbone_short + '-' + self.model_opt
        self.model_path = prepare_saved_model(model_name)
        self.same_domain_inds_path = os.path.join(os.path.dirname(self.model_path), 'train-val-tst-idx_dict.json')


    def set_valid_opts(self):

        self.all_methods = ['diffusion', 'knn',
                            ]

        self.n_gpu = torch.cuda.device_count()
        self.metric = 'L2'
        self.batch_size = 32
        self.batch_size_pred = 32 # smaller batch size works better with large num workers!!
        self.ref_mode = 'flex_reference'
        self.ref_point_num = self.class_number # same as number of ranks
        self.start_norm = True
        self.num_workers = 26
        self.predict_scheme = 'weighted-mean'
        self.reciprocal = False
        self.reciprocal_k1 = 20
        self.reciprocal_k2 = 10  # 3
        self.reciprocal_l = 0.5 # weight of original distance
        self.logfile = None
        self.test_for_same_domain = False # use cfg.ini_knn_k
        self.alpha_expansion = False
        self.alpha = 3
        self.alpha_n = 10 # 3

        if self.test_for_same_domain:
            self.init_knn_k = 10
        # k for knn
        self.adjust_feature_dim = 512  # no feature reduction
        self.test_for_cross_domain = True
        self.sample_equal_infer = False # sample equal number of samples from each rank for infer data
        self.infer_num_per_class = 100
        self.use_reduced_feature = True
        # infer mode

        if not self.few_shot:
            self.init_knn_k = 10
            self.sample_diffusion = 100
            self.alpha_expansion = False
            self.alpha = 3
            self.alpha_n = 10  # 3
            self.diffuse_alpha = 0.99
            self.diffuse_gamma = 3
            self.diffuse_k = 10  # k nearest neighbors for constructing the graph
            self.diffuse_k_value = 10
            self.reduce_dim_one = False

        if self.few_shot:
            self.few_shot_random_repeat = 10 # repeat random sampling for few shot and report mean, std
            self.few_shot_head_type = 'all' # 'linear', 'rf', 'dense', 'prototypical', 'diffusion
            self.reduce_dim_one = False # reduce embedding dimension to 1 and performance regression
            self.alpha = 3
            self.alpha_n = self.few_shot_num
            self.diffuse_alpha = 0.99
            self.diffuse_gamma = 3
            self.diffuse_proto_eta = 0.9 # weight of original embedding
            self.diffusion_value_use_entropy = False # entropy or z value directly
            self.diffuse_k_adap = False # whether using percentile to adapt k for affinity graph construction
            self.diffuse_calibrate = False
            if self.diffuse_k_adap: # percenilte
                self.diffuse_k = 99
            else:
                self.diffuse_k = self.few_shot_num
            self.proto_build = 'mean'
            self.spectral_emb_k = self.few_shot_num
            self.spectral_emb_njobs  = 1

