import argparse
import os
import sys

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import f1_score

from networks.util import prepare_model
from utils.diffusion import alpha_query_expansion
from utils.plot_utils import *
from utils.plot_utils import plot_scatter
from utils.util import AverageMeter, extract_embs
from utils.util import prepare_few_shot_data, query_knn_head, query_linear_fitting, \
    query_diffusion_head, process_embs_fixed_k_search, prepare_few_shot_data_regressor, query_regressor_linear_calib, \
    drr


def main(cfg):
    # validation on cross-domain (cross area) datasets
    # print('=========================== Source data: ', cfg.source_dataset)
    # print('=========================== Cross domain infer data: ', cfg.infer_dataset)
    print('=========================== Target variable: ', cfg.target)

    from data.get_dataset_adapt import get_datasets
    loader_dict = get_datasets(cfg)
    cfg.n_ranks = cfg.class_number
    model = prepare_model(cfg)

    if cfg.model_path:
        model.load_state_dict(torch.load(cfg.model_path)['model'])
        # print(f'[*] ######################### model loaded from {cfg.model_path}')

    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    # print('==> validation...')
    if cfg.model == 'GOL':
        r2_dff, r2_knn, rmse_dff, rmse_knn, std_r2_dff, std_r2_knn, std_rmse_dff, std_rmse_knn = validate(loader_dict, model, cfg)
        # print('[*] Validation ends')
        return r2_dff, r2_knn, rmse_dff, rmse_knn, std_r2_dff, std_r2_knn, std_rmse_dff, std_rmse_knn
    elif cfg.model == 'regressor':
        mean_r2, std_r2, mean_rmse, std_rmse, no_adapt_r2, no_adapt_rmse = validate(loader_dict, model, cfg)
        # print('[*] Validation ends')
        return mean_r2, std_r2, mean_rmse, std_rmse, no_adapt_r2, no_adapt_rmse



def validate(loader_dict, model, cfg, lim = 30):

    model.eval()

    if cfg.test_for_same_domain:
        print('=========================== test for same domain')
        if cfg.model == 'GOL':
            print('=============================================================================')
            print('test scheme BASE: using ref data from same domain using knn embedding search')
            data_time = AverageMeter()
            embs_train, train_labels, train_ranks, train_img_ids = extract_embs(model.encoder, loader_dict['train_for_test'],
                                                                 return_labels=True, return_img_id=True)
            embs_train = embs_train.cuda()
            embs_test, test_labels, test_ranks, test_img_ids = extract_embs(model.encoder, loader_dict['test_same_domain'],
                                                                            return_labels=True, return_img_id=True)
            embs_test = embs_test.cuda()
            n_test = len(embs_test)
            n_batch = int(np.ceil(n_test / cfg.batch_size))

            # vanilla kNN / k-reciprocal re-ranking kNN
            if cfg.alpha_expansion:
                embs_test = alpha_query_expansion(embs_test, alpha=cfg.alpha, n=cfg.alpha_n)
                # embs_train = alpha_query_expansion(embs_train, alpha=cfg.alpha, n=cfg.alpha_n)


            preds_all, preds_rank_all = process_embs_fixed_k_search(embs_test, embs_train, train_labels, train_ranks,
                                                                    data_time, n_test, n_batch, cfg,
                                                                    rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                                    rec_l=cfg.reciprocal_l)


            sys.stdout.flush()


            if cfg.target == 'count':
                lim = 200
            elif cfg.target == 'height':
                lim = 30
            elif cfg.target == 'treecover5m':
                lim = 1

            if cfg.logscale:
                test_labels = np.exp(test_labels) - cfg.offset
                preds_all = np.exp(preds_all) - cfg.offset

            plot_scatter(test_labels, preds_all, 'Predicted vs. target', 'Reference',
                         'Predicted'
                         , lim, font=35, spinexy=0, markersize=30, xtic=1, showr2=1, showfit=0)

            print('test r2: ', r2_score(test_labels, preds_all))
            print('test rank f1: ', f1_score(test_ranks, preds_rank_all, average='micro')) # micro for imbalance class
            print('test mae', np.mean(np.abs(test_labels - preds_all)))
            print('test rmae', np.mean(np.abs(test_labels - preds_all)) / np.mean(test_labels))


        elif cfg.model == 'regressor':
            # direct estimation
            print('=============================================================================')
            print('test scheme BASE: direct estimation with final linear layer')
            preds_all = []
            targets_all = []
            target_ranks_all = []
            for idx, (x_base, cur_target, cur_rank, _) in enumerate(loader_dict['test_same_domain']):
                if torch.cuda.is_available():
                    x_base = x_base.float().cuda()
                with torch.no_grad():
                    output = model.encoder(x_base).squeeze()
                preds_all.append(output.cpu().detach().numpy())
                targets_all.append(cur_target)
                target_ranks_all.append(cur_rank)
            preds_all = np.concatenate(preds_all, axis=0)
            targets_all = np.concatenate(targets_all, axis=0)

            print('test r2: ', r2_score(targets_all, preds_all))
            print('test mae: ', np.mean(np.abs(targets_all - preds_all)))

    if cfg.test_for_cross_domain:
        # print('================= test for cross domain')
        embs_test, test_labels, test_ranks, test_img_ids = extract_embs(model.encoder, loader_dict['test_cross_domain'],
                                                                        return_labels=True, return_img_id=True, use_reduced_feature = cfg.use_reduced_feature)
        embs_test = embs_test.cuda()
        n_test = len(embs_test)


        if not cfg.few_shot:
            if cfg.model == 'GOL':

                # vanilla kNN / k-reciprocal re-ranking kNN
                print('=============================================================================')
                print('test scheme BASE: using ref data from cross domain using knn embedding search')
                embs_train, train_labels, train_ranks, train_img_ids = extract_embs(model.encoder, loader_dict['train_for_test'],
                                                                     return_labels=True, return_img_id=True)
                embs_train = embs_train.cuda()

                mae, rmae, r2, f1 = query_knn_head(embs_test, embs_train, train_labels, train_ranks, test_labels, test_ranks, cfg, lim)
                print('knn rank f1: ', f1)
                print('knn r2 score: ', r2)

                embs_test_alpha = alpha_query_expansion(embs_test, alpha=cfg.alpha, n=cfg.alpha_n)
                mae, rmae, r2, f1 = query_knn_head(embs_test_alpha, embs_train, train_labels, train_ranks, test_labels, test_ranks, cfg, lim)
                print('alpha expansion rank f1: ', f1)
                print('alpha expansion r2 score: ', r2)
                print('=============================================================================')
                print('testing scheme trying diffusion')
                print('sample part of ref data for diffusion')
                # repeat 10 times and report mean std of r2 and f1
                r2ss = []
                f1ss = []
                r2_lss = []
                for i in range(5):
                    np.random.seed(i)
                    # saturated sampling from each rank
                    idx = []
                    for cc in range(cfg.class_number):
                        idx.append(np.random.choice(np.where(train_ranks == cc)[0],
                                                    int(cfg.sample_diffusion/cfg.class_number), replace=False))
                    idx = np.concatenate(idx, axis=0)
                    sub_embs_train = embs_train[idx]
                    sub_train_labels = train_labels[idx]
                    sub_train_ranks = train_ranks[idx]
                    mae, rmae, r2, f1 = query_diffusion_head(embs_test, sub_embs_train, sub_train_labels,
                                                             sub_train_ranks, test_labels, test_ranks, cfg, lim)

                    mae, rmae, r2_linear, preds = query_linear_fitting(embs_test, sub_embs_train, sub_train_labels,
                                                                test_labels, cfg, lim)

                    r2_lss.append(r2_linear)

                    r2ss.append(r2)
                    f1ss.append(f1)
                print('diffusion rank f1: ', np.mean(f1ss), np.std(f1ss))
                print('diffusion r2 score: ', np.mean(r2ss), np.std(r2ss))
                print('linear r2 score: ', np.mean(r2_lss), np.std(r2_lss))

            elif cfg.model == 'regressor':
                # direct estimation
                print('=============================================================================')
                print('test scheme BASE: direct estimation with final linear layer')
                preds_all = []
                targets_all = []
                for idx, (x_base, cur_target, cur_rank, _) in enumerate(loader_dict['test_cross_domain']):
                    if torch.cuda.is_available():
                        x_base = x_base.float().cuda()
                    with torch.no_grad():
                        output = model.encoder(x_base).squeeze()
                    preds_all.append(output.cpu().detach().numpy())
                    targets_all.append(cur_target)

                targets_all = np.concatenate(targets_all, axis=0)
                preds_all = np.concatenate(preds_all, axis=0)

                print('test r2: ', r2_score(targets_all, preds_all))
                print('test mae: ', np.mean(np.abs(targets_all - preds_all)))



        elif cfg.few_shot: # some query data with known labels
            if cfg.model == 'GOL' or cfg.model == 'GOLvanilla':
                r2_df = pd.DataFrame(index=range(cfg.few_shot_random_repeat), columns=cfg.all_methods)
                rmse_df = pd.DataFrame(index=range(cfg.few_shot_random_repeat), columns=cfg.all_methods)
                embs_test_np = embs_test.cpu().detach().numpy()

                from sklearn import cluster
                sc = cluster.SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_init=100, n_jobs=1)
                embs_test_cluster = sc.fit_predict(embs_test_np)
                cents = []
                cent_ranks = []
                for i in range(cfg.class_number):
                    cents.append(np.nanmean(embs_test_np[embs_test_cluster == i], axis=0))
                    cent_ranks.append(i)
                cents = np.array(cents)
                cent_ranks = np.array(cent_ranks)
                value = drr(embs_test_np, cents, cent_ranks, beta=0.5)
                print('============================')
                print('source data', cfg.source_dataset)
                print('infer data', cfg.infer_dataset)
                # ipdb.set_trace()
                print('Drr value: ', value)


                for itr in range(cfg.few_shot_random_repeat):
                    # print('===================== Random sampling repeat: ', itr)
                    # print('number of total shot: ', cfg.few_shot_num * cfg.class_number)
                    # fix seed
                    np.random.seed(itr)

                    embs_test_test, labels_test_test, ranks_test_test, embs_test_ref, labels_test_ref, ranks_test_ref, ref_idx = prepare_few_shot_data(
                        embs_test, test_labels, test_ranks, cfg)


                    mae, rmse, r2, _ = query_knn_head(embs_test_test, embs_test_ref, labels_test_ref,
                                                       ranks_test_ref, labels_test_test, ranks_test_test, cfg, lim)
                    r2_df.loc[itr, 'knn'] = r2
                    rmse_df.loc[itr, 'knn'] = rmse


                    mae, rmse, r2, _, cur_preds, cur_labels, diffuser = query_diffusion_head(embs_test_test, embs_test_ref, labels_test_ref,
                                                                ranks_test_ref, labels_test_test, ranks_test_test, cfg, lim,
                                                                                    return_preds = True,
                                                                                    return_diffuser=True)

                    r2_df.loc[itr, 'diffusion'] = r2
                    rmse_df.loc[itr, 'diffusion'] = rmse

                r2_df.loc['mean'] = r2_df.mean(axis=0)
                r2_df.loc['std'] = r2_df.std(axis=0)
                rmse_df.loc['mean'] = rmse_df.mean(axis=0)
                rmse_df.loc['std'] = rmse_df.std(axis=0)

                return r2_df.loc['mean']['diffusion'], r2_df.loc['mean']['knn'], rmse_df.loc['mean']['diffusion'], rmse_df.loc['mean']['knn'], \
                          r2_df.loc['std']['diffusion'], r2_df.loc['std']['knn'], rmse_df.loc['std']['diffusion'], rmse_df.loc['std']['knn']


            elif cfg.model == 'regressor':

                preds_all = []
                targets_all = []
                ranks_all = []
                for idx, (x_base, cur_target, cur_rank, _) in enumerate(loader_dict['test_cross_domain']):
                    if torch.cuda.is_available():
                        x_base = x_base.float().cuda()
                    with torch.no_grad():
                        output = model.encoder(x_base).squeeze()
                    preds_all.append(output.cpu().detach().numpy())
                    targets_all.append(cur_target)
                    ranks_all.append(cur_rank)

                targets_all = np.concatenate(targets_all, axis=0)
                preds_all = np.concatenate(preds_all, axis=0)
                ranks_all = np.concatenate(ranks_all, axis=0)
                no_apat_r2 = r2_score(targets_all, preds_all)
                no_apat_rmse = np.mean(np.abs(targets_all - preds_all))
                r2ss = []
                rmsess = []
                for itr in range(cfg.few_shot_random_repeat):

                    np.random.seed(itr)
                    preds_test, labels_test, ranks_test, preds_ref, labels_ref, ranks_ref, test_ref_ind = prepare_few_shot_data_regressor(
                                                                                                        preds_all, targets_all, ranks_all, cfg)
                    # using ref to calibrate with linear regression
                    r2, rmse = query_regressor_linear_calib(preds_test, labels_test, preds_ref, labels_ref)
                    r2ss.append(r2)
                    rmsess.append(rmse)

                return np.mean(r2ss), np.std(r2ss), np.mean(rmsess), np.std(rmsess), no_apat_r2, no_apat_rmse


            else:
                raise NotImplementedError





if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', type=str, default='DK', help='source dataset')
    parser.add_argument('--backbone', type=str, default='vit_b16_reduce', help='backbone')
    parser.add_argument('--target', type=str, default='height', help='target variable')
    args = parser.parse_args()

    from config.cross_domain_fewshot import Config
    source_list = ['france']
    infer_list = ['SP', 'slovakia', 'slovenia']
    few_shot_list = [5]

    for few_shot_num in few_shot_list:
        r2_dffs = []
        r2_dffs_stds = []
        r2_knns = []
        r2_knns_stds = []
        rmse_dffs = []
        rmse_dffs_stds = []
        rmse_knns = []
        rmse_knns_stds = []
        r2ss = []  # for regressor
        stdss = []
        noadapt_r2s = []
        rmsess = []
        rmse_stds = []
        noadapt_rmses = []
        for source_dataset in source_list:
            for infer_dataset in infer_list:

                cfg = Config(source_dataset, infer_dataset, args.target, args.backbone, few_shot_num)
                if cfg.model == 'GOL' or cfg.model == 'GOLvanilla':
                    mean_difu, mean_knn, mean_difu_rmse, mean_knn_rmse, \
                        std_dffu, std_knn, std_difu_rmse, std_knn_rmse = main(cfg)
                    r2_dffs.append(mean_difu)
                    r2_dffs_stds.append(std_dffu)
                    r2_knns.append(mean_knn)
                    r2_knns_stds.append(std_knn)
                    rmse_dffs.append(mean_difu_rmse)
                    rmse_dffs_stds.append(std_difu_rmse)
                    rmse_knns.append(mean_knn_rmse)
                    rmse_knns_stds.append(std_knn_rmse)
                elif cfg.model == 'regressor':
                    mean_r2, std_r2, mean_rmse, std_rmse, noadapt_r2, no_adapt_rmse = main(cfg)
                    r2ss.append(mean_r2)
                    stdss.append(std_r2)
                    rmsess.append(mean_rmse)
                    rmse_stds.append(std_rmse)
                    noadapt_r2s.append(noadapt_r2)
                    noadapt_rmses.append(no_adapt_rmse)

        print('=========================== Few shot num: ', few_shot_num)

        if cfg.model == 'GOL' or cfg.model == 'GOLvanilla':
            print('********************************R2********************************')
            print('======== Diffusion: r2 for all querys ============')
            print('r2s', r2_dffs)
            print('stds of r2' , r2_dffs_stds)
            # ipdb.set_trace()
            print('mean r2 of Diffusion across query sets: ', np.mean(r2_dffs))
            print('std of r2 of Diffusion across query sets: ', np.std(r2_dffs))
            print('======== knn: r2 for all querys ============')
            print('r2s', r2_knns)
            print('stds of r2', r2_knns_stds)
            print('mean r2 of knn across query sets: ', np.mean(r2_knns))
            print('std of r2 of knn across query sets: ', np.std(r2_knns))


        elif cfg.model == 'regressor':
            print('********************************R2********************************')
            print('======= before calibration r2 for all querys ======')
            print('no adapt r2s', noadapt_r2s)
            print('mean of no apat r2= ', np.mean(noadapt_r2s))
            print('std of no apat r2= ', np.std(noadapt_r2s))
            print('======= after calibration r2 for all querys ======')
            print('r2s', r2ss)
            print('stds', stdss)
            print('mean r2 of regressor across query sets: ', np.mean(r2ss))
            print('std of r2 of regressor across query sets: ', np.std(r2ss))



