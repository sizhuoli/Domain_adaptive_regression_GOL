import glob
import pickle
import pandas as pd
import numpy as np
import rasterio
import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from data.datasets import Base_source
import ipdb
import json
import matplotlib.pyplot as plt
import time
def get_datasets(conf):
    """

    path_imgs: list of image paths

    path_targets: list of targets

    """

    df = pd.read_csv(conf.label_path)
    df = df[df[conf.statis] <= conf.upper_clip]
    img_files = df[conf.img_id].tolist()
    # get full image path
    img_files = [conf.img_base + f for f in img_files]
    labels = df[conf.statis].tolist()
    assert len(img_files) == len(labels), 'number of image files and label do not match'
    print('label: ', conf.statis)

    if conf.class_interval_scheme == 'balanced':
        raise NotImplementedError
    elif conf.class_interval_scheme == 'numeric':
        print('using equal split of ranks: upper label and lower buond: ', conf.upper_bound, conf.lower_bound)
        ranges = np.linspace(conf.lower_bound, conf.upper_bound, conf.class_number + 1)

    ranges[0] = 0
    ranges[-1] = 1000  # upper bound
    print('label ranges: ', ranges)
    ranks = np.digitize(labels, ranges) - 1

    print('=====> rank frequency: ', np.unique(ranks, return_counts=True))

    # randomly split into train and val
    # set seed
    if conf.set_random_seed:
        np.random.seed(0)

    # split into train, val, test with given ratio
    train_idx = np.random.choice(len(img_files), int(len(img_files) * conf.train_ratio), replace=False)
    val_idx = np.random.choice(np.setdiff1d(np.arange(len(img_files)), train_idx), int(len(img_files) * conf.val_ratio), replace=False)
    test_idx = np.setdiff1d(np.arange(len(img_files)), np.concatenate([train_idx, val_idx]))

    # split into train and val
    train_imgs = [img_files[i] for i in train_idx]
    train_targets = [labels[i] for i in train_idx]
    train_ranks = [ranks[i] for i in train_idx]
    val_imgs = [img_files[i] for i in val_idx]
    val_targets = [labels[i] for i in val_idx]
    val_ranks = [ranks[i] for i in val_idx]
    test_imgs = [img_files[i] for i in test_idx]
    test_targets = [labels[i] for i in test_idx]
    test_ranks = [ranks[i] for i in test_idx]

    assert len(np.intersect1d(train_imgs, val_imgs)) == 0
    assert len(np.intersect1d(train_imgs, test_imgs)) == 0
    assert len(np.intersect1d(val_imgs, test_imgs)) == 0

    # save idx to json
    idx_dict = dict()
    idx_dict['train_idx'] = train_idx.tolist()
    idx_dict['val_idx'] = val_idx.tolist()
    idx_dict['test_idx'] = test_idx.tolist()
    with open(conf.save_folder + '/train-val-tst-idx_dict.json', 'w') as f:
        json.dump(idx_dict, f)


    print('====> train ranks: ', np.unique(train_ranks, return_counts=True))
    print('====> val ranks: ', np.unique(val_ranks, return_counts=True))
    print('====> test ranks: ', np.unique(test_ranks, return_counts=True))
    print('label percentile: ', np.percentile(labels, [0, 25, 50, 75, 100]))
    train_sample_weights = np.ones(len(train_imgs))

    loader_dict = dict()
    if conf.model == 'GOL':
        loader_dict['train'] = DataLoader(Base_source.Train(conf, train_imgs, train_targets,
                                                             train_sample_weights, train_ranks, tau = conf.tau,
                                                             logscale=conf.logscale, is_filelist=conf.is_filelist),
                                          batch_size=conf.batch_size, shuffle=True, drop_last=True,
                                          num_workers=conf.num_workers, pin_memory=True)
        # training embedding for validation
        loader_dict['train_for_val'] = DataLoader(Base_source.Valid(conf, train_imgs, train_targets, train_ranks),
                                          batch_size=conf.batch_size, shuffle=False, drop_last=False,
                                                  num_workers=conf.num_workers, pin_memory=True)
    elif conf.model == 'regressor':
        loader_dict['train'] = DataLoader(Base_source.Train_vanilla(conf, train_imgs, train_targets),
                                          batch_size=conf.batch_size, shuffle=True, drop_last=True,
                                          num_workers=conf.num_workers, pin_memory=True)
    # validation data also randomly cropped
    loader_dict['val'] = DataLoader(Base_source.Valid(conf, val_imgs, val_targets, val_ranks),
                                      batch_size=conf.batch_size, shuffle=False, drop_last=False,
                                    num_workers=conf.num_workers, pin_memory=True)

    loader_dict['test'] = DataLoader(Base_source.Valid(conf, test_imgs, test_targets, test_ranks),
                                        batch_size=conf.batch_size, shuffle=False, drop_last=False,
                                        num_workers=conf.num_workers, pin_memory=True)


    return loader_dict



