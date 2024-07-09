import glob
import pickle
import pandas as pd
import numpy as np
import rasterio
import tqdm
from torch.utils.data import DataLoader
from data.datasets import Adapt_target

def get_datasets(conf, mode = 'hyperselect'):
    """

    """
    # loading validation data
    print('loading infer data')
    label_df = pd.read_csv(conf.infer_label_path)
    label_df[conf.fileid] = \
    label_df[conf.imagepath].str.split('/').str[-1].str.split('fileID_').str[-1].str.split('_').str[0
    ]
    print('raw label_df of length : ', len(label_df))
    # remove rows with nan imagepath
    label_df = label_df[~label_df[conf.imagepath].isna()]
    l1 = len(label_df)
    print('after nan imagepath filter : ', l1)

    # remove wrong data
    if conf.wrongFileRemove:
        with open(conf.wrongFilePath, 'rb') as fn:
            list_wrong = pickle.load(fn)

        label_df = label_df[~label_df[conf.fileid].isin(list_wrong)]
        print('after wrong data filter : ', len(label_df))
        print('percent of data removed after wrong data filtering : ', (l1 - len(label_df)) / l1 * 100, '%')

    if '/' not in label_df[conf.imagepath].iloc[0]:
        label_df[conf.imagepath] = conf.img_dir + label_df[conf.imagepath]

    img_list = label_df[conf.imagepath].unique()

    if conf.img_filter_check:
        img_list2 = []
        # check if min of image is 0 (then image is not complete, remove from list)
        for f in tqdm.tqdm(img_list):
            with rasterio.open(f) as src:
                if np.min(src.read()) > 0:
                    img_list2.append(f)
        print('%d percent images removed after filter' % ((len(img_list) - len(img_list2))/len(img_list) * 100))
        print('number of images after filter: ', len(img_list2))
        img_list = img_list2

    label_df = label_df[label_df[conf.imagepath].isin(img_list)]

    if conf.class_interval_scheme == 'numeric':
        # get percentiles of attribute
        lw = np.percentile(label_df[conf.attribute], conf.infer_lower_bound)
        up = np.percentile(label_df[conf.attribute], conf.infer_upper_bound)
        infer_ranges = np.linspace(lw, up, conf.infer_class_number + 1)
    elif conf.class_interval_scheme == 'balanced':
        infer_ranges = np.percentile(label_df[conf.attribute], np.linspace(0, 100, conf.infer_class_number + 1))
    else:
        raise NotImplementedError

    infer_ranges[0] = 0
    infer_ranges[-1] = 10000

    print('class interval scheme : ', conf.class_interval_scheme)
    print('============== ranges for infer data: ', infer_ranges)

    label_list = [label_df[label_df[conf.imagepath] == img][conf.attribute].values[0] for img in img_list]
    rank_list = [np.digitize(lb, infer_ranges) - 1 for lb in label_list]
    print('number of samples in each rank : ', np.unique(rank_list, return_counts=True))


    if conf.sample_equal_infer:
        infer_img_list = []
        sampled_idx = []
        # for infer data: sample fixed number of samples from each rank to balance data
        for i in range(conf.infer_class_number):
            idx = np.where(np.array(rank_list) == i)[0]

            sampled_idx += list(np.random.choice(idx, conf.infer_num_per_class, replace=False))

        img_list = [img_list[j] for j in sampled_idx]
        label_list = [label_list[j] for j in sampled_idx]
        rank_list = [rank_list[j] for j in sampled_idx]

        print('number of samples in each rank after sampling : ', np.unique(rank_list, return_counts=True))

    # sample * samples in each rank into train
    train_list = []
    val_list = []
    test_list = []
    train_label_list = []
    val_label_list = []
    test_label_list = []
    train_rank_list = []
    val_rank_list = []
    test_rank_list = []
    for i in range(conf.infer_class_number):
        idx = np.where(np.array(rank_list) == i)[0]
        idx_train = np.random.choice(idx, conf.few_shot_num, replace=False)

        if mode == 'hyperselect':
            # subset train into adapt and val
            idx_val = np.random.choice(idx_train, conf.few_shot_valid_num, replace=False)
            idx_adapt = np.setdiff1d(idx_train, idx_val)
            val_list.extend(list(np.array(img_list)[idx_val]))
            val_label_list.extend(list(np.array(label_list)[idx_val]))
            val_rank_list.extend(list(np.array(rank_list)[idx_val]))

        else:
            idx_adapt = idx_train

        train_list.extend(list(np.array(img_list)[idx_adapt]))
        idx_test = np.setdiff1d(idx, idx_train)
        test_list.extend(list(np.array(img_list)[idx_test]))
        train_label_list.extend(list(np.array(label_list)[idx_adapt]))
        test_label_list.extend(list(np.array(label_list)[idx_test]))
        train_rank_list.extend(list(np.array(rank_list)[idx_adapt]))
        test_rank_list.extend(list(np.array(rank_list)[idx_test]))

    # check if train and val are overlapping
    if mode == 'hyperselect':
        # ipdb.set_trace()
        assert len(set(train_list) & set(val_list)) == 0
        assert len(set(val_list) & set(test_list)) == 0
        print('val list length : ', len(val_list))

    # check if train and val are overlapping
    assert len(set(train_list) & set(test_list)) == 0
    print('train list length : ', len(train_list))
    print('test list length : ', len(test_list))

    loader_dict = dict()

    if conf.train_loss_scheme == 'gol':
        loader_dict['train'] = DataLoader(Adapt_target.Train(conf, train_list, train_label_list, train_rank_list),
                                            batch_size=conf.batch_size_pred, shuffle=True, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

        loader_dict['train_for_val'] = DataLoader(Adapt_target.Valid(conf, train_list, train_label_list, train_rank_list),
                                          batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

        if mode == 'hyperselect':

            loader_dict['val'] = DataLoader(Adapt_target.Valid(conf, val_list, val_label_list, val_rank_list),
                                              batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

        loader_dict['test'] = DataLoader(Adapt_target.Valid(conf, test_list, test_label_list, test_rank_list),
                                            batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)


    else:
        raise NotImplementedError

    return loader_dict





