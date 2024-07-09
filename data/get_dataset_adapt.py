import glob
import pickle
import pandas as pd
import numpy as np
import rasterio
import tqdm
from torch.utils.data import DataLoader
from data.datasets import Base_source
from data.datasets import Adapt_target
import json

def get_datasets(conf):
    """

    """
    # print('==================== loading training data for inference')
    df = pd.read_csv(conf.label_path)
    df = df[df[conf.statis] <= conf.upper_clip]
    if 'cover' in conf.statis:
        df[conf.statis] = df[conf.statis] * conf.rescale
    if conf.normalize_target:
        df[conf.statis] = (df[conf.statis] - df[conf.statis].min()) / (df[conf.statis].max() - df[conf.statis].min())

    img_files = df[conf.img_id].tolist()
    labels = df[conf.statis].tolist()
    assert len(img_files) == len(labels), 'number of image files and label do not match'
    if conf.class_interval_scheme == 'balanced':
        raise NotImplementedError
    elif conf.class_interval_scheme == 'numeric':
        ranges = np.linspace(conf.lower_bound, conf.upper_bound, conf.class_number + 1)

    ranges[0] = 0
    ranges[-1] = 1000  # upper bound
    ranks = np.digitize(labels, ranges) - 1

    # for same domain test, load train, val, test ind from dict json
    with open(conf.same_domain_inds_path, 'r') as fn:
        same_domain_inds = json.load(fn)
        train_idx = same_domain_inds['train_idx']
        test_idx = same_domain_inds['test_idx']

    assert len(np.intersect1d(train_idx, test_idx)) == 0, 'train and test overlap'

    train_imgs = [img_files[i] for i in train_idx]
    train_targets = [labels[i] for i in train_idx]
    train_ranks = [ranks[i] for i in train_idx]
    test_imgs = [img_files[i] for i in test_idx]
    test_targets = [labels[i] for i in test_idx]
    test_ranks = [ranks[i] for i in test_idx]

    # # loading validation data
    # print('=================== loading cross domain infer data from {}'.format(conf.infer_label_path))
    label_df = pd.read_csv(conf.infer_label_path)
    label_df = label_df[label_df[conf.attribute] <= conf.infer_upper_clip]
    if 'cover' in conf.attribute:
        label_df[conf.attribute] = label_df[conf.attribute] * conf.rescale

    if conf.normalize_target:
        label_df[conf.attribute] = (label_df[conf.attribute] - label_df[conf.attribute].min()) / (label_df[conf.attribute].max() - label_df[conf.attribute].min())

    label_df = label_df[~label_df[conf.imagepath].isna()]
    l1 = len(label_df)

    if conf.wrongFileRemove:
        # assign substring of imagepath to fileid
        label_df[conf.fileid] = \
            label_df[conf.imagepath].str.split('/').str[-1].str.split('fileID_').str[-1].str.split('_').str[0
            ]
        with open(conf.wrongFilePath, 'rb') as fn:
            list_wrong = pickle.load(fn)

        label_df = label_df[~label_df[conf.fileid].isin(list_wrong)]
        # print('after wrong data filter : ', len(label_df))
        # print('percent of data removed after wrong data filtering : ', (l1 - len(label_df)) / l1 * 100, '%')

    # check if full path or basename only
    if '/' not in label_df[conf.imagepath].iloc[0]:
        label_df[conf.imagepath] = conf.img_dir + label_df[conf.imagepath]

    infer_img_list = label_df[conf.imagepath].unique()

    if conf.img_filter_check:
        infer_img_list2 = []
        # check if min of image is 0 (then image is not complete, remove from list)
        for f in tqdm.tqdm(infer_img_list):
            with rasterio.open(f) as src:
                if np.min(src.read()) > 0:
                    infer_img_list2.append(f)
        # print('%d percent images removed after image validity filter' % ((len(infer_img_list) - len(infer_img_list2))/len(infer_img_list) * 100))
        print('number of images after image validity filter: ', len(infer_img_list2))
        infer_img_list = infer_img_list2

    # infer ranges
    if conf.class_interval_scheme == 'numeric':
        lw = np.percentile(label_df[conf.attribute], conf.infer_lower_bound)
        up = np.percentile(label_df[conf.attribute], conf.infer_upper_bound)
        infer_ranges = np.linspace(lw, up, conf.infer_class_number + 1)
    elif conf.class_interval_scheme == 'balanced':
        infer_ranges = np.percentile(label_df[conf.attribute], np.linspace(0, 100, conf.infer_class_number + 1))
    else:
        raise NotImplementedError

    infer_ranges[0] = 0
    infer_ranges[-1] = 1000  # upper bound
    label_list = [label_df[label_df[conf.imagepath] == img][conf.attribute].values[0] for img in infer_img_list]
    rank_list = [np.digitize(lb, infer_ranges) - 1 for lb in label_list]

    if conf.sample_equal_infer:
        sampled_idx = []
        for i in range(conf.infer_class_number):
            idx = np.where(np.array(rank_list) == i)[0]
            sampled_idx += list(np.random.choice(idx, conf.infer_num_per_class, replace=False))

        infer_img_list = [infer_img_list[j] for j in sampled_idx]
        label_list = [label_list[j] for j in sampled_idx]
        rank_list = [rank_list[j] for j in sampled_idx]

    loader_dict = dict()
    # training embedding for infer
    loader_dict['train_for_test'] = DataLoader(Base_source.Valid(conf, train_imgs, train_targets, train_ranks),
                                              batch_size=conf.batch_size, shuffle=False, drop_last=False,
                                              num_workers=conf.num_workers, pin_memory=True)

    loader_dict['test_same_domain'] = DataLoader(Base_source.Valid(conf, test_imgs, test_targets, test_ranks),
                                    batch_size=conf.batch_size_pred, shuffle=False, drop_last=False,
                                    num_workers=conf.num_workers, pin_memory=True)


    loader_dict['test_cross_domain'] = DataLoader(Adapt_target.Valid(conf, infer_img_list, label_list, rank_list),
                                      batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)


    return loader_dict
