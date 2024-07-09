import glob
import pickle
import pandas as pd
import numpy as np
import rasterio
import tqdm
import torch
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
    label_df = label_df[~label_df[conf.imagepath].isna()]
    l1 = len(label_df)
    print('after nan imagepath filter : ', l1)
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
        sampled_idx = []
        for i in range(conf.infer_class_number):
            idx = np.where(np.array(rank_list) == i)[0]
            sampled_idx += list(np.random.choice(idx, conf.infer_num_per_class, replace=False))
        img_list = [img_list[j] for j in sampled_idx]
        label_list = [label_list[j] for j in sampled_idx]
        rank_list = [rank_list[j] for j in sampled_idx]

        print('number of samples in each rank after sampling : ', np.unique(rank_list, return_counts=True))

    # sample * samples in each rank into train
    train_list = []
    test_list = []
    val_list = []
    train_label_list = []
    test_label_list = []
    val_label_list = []

    for i in range(conf.infer_class_number):
        idx = np.where(np.array(rank_list) == i)[0]
        idx_train = np.random.choice(idx, conf.few_shot_num, replace=False)

        if mode == 'hyperselect':
            # subset train into adapt and val
            idx_val = np.random.choice(idx_train, conf.few_shot_valid_num, replace=False)
            idx_adapt = np.setdiff1d(idx_train, idx_val)
            val_list.extend(list(np.array(img_list)[idx_val]))
            val_label_list.extend(list(np.array(label_list)[idx_val]))
        else:
            idx_adapt = idx_train

        train_list.extend(list(np.array(img_list)[idx_adapt]))
        idx_test = np.setdiff1d(idx, idx_train)
        test_list.extend(list(np.array(img_list)[idx_test]))
        train_label_list.extend(list(np.array(label_list)[idx_adapt]))
        test_label_list.extend(list(np.array(label_list)[idx_test]))

    # check if train and val are overlapping
    if mode == 'hyperselect':
        # ipdb.set_trace()
        assert len(set(train_list) & set(val_list)) == 0
        assert len(set(val_list) & set(test_list)) == 0
        print('val list length : ', len(val_list))

    assert len(set(train_list) & set(test_list)) == 0

    print('train list length : ', len(train_list))

    print('test list length : ', len(test_list))

    # ipdb.set_trace()
    loader_dict = dict()

    loader_dict['train'] = DataLoader(Adapt_target.Train_vanilla(conf, train_list, train_label_list),
                                        batch_size=conf.batch_size_pred, shuffle=True, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

    if mode == 'hyperselect':
        loader_dict['val'] = DataLoader(Adapt_target.Valid_vanilla(conf, val_list, val_label_list),
                                      batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

    loader_dict['test'] = DataLoader(Adapt_target.Valid_vanilla(conf, test_list, test_label_list),
                                        batch_size=conf.batch_size_pred, shuffle=False, drop_last=False, num_workers=conf.num_workers, pin_memory=True)

    return loader_dict


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        sampler = torch.utils.data.RandomSampler(dataset,
            replacement=True)


        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch



