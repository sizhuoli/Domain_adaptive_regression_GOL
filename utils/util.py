import os
# ignore warning
import warnings
from datetime import datetime

import cv2
import numpy as np
import rasterio
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, r2_score, f1_score

warnings.filterwarnings("ignore")
from torchinfo import summary
import ipdb
import time
from sklearn.manifold import TSNE, SpectralEmbedding

from utils.comparison_utils import find_kNN
from collections import defaultdict
from matplotlib.patches import Rectangle

from utils.diffusion import Diffuser
from utils.plot_utils import plot_emb
from sklearn.linear_model import LinearRegression


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassWiseAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.val = np.zeros([self.n_cls,])
        self.avg = np.zeros([self.n_cls,])
        self.sum = np.zeros([self.n_cls,])
        self.count = np.ones([self.n_cls,]) * 1e-7
        self.total_avg = 0

    def update(self, val, n=[1,1,1]):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.total_avg = np.sum(self.sum) / np.sum(self.count)


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cls_accuracy(output, target, n_cls=3):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1)
        correct = pred.eq(target).cpu().numpy()
        accs = np.zeros([n_cls,])
        cnts = np.ones([n_cls,]) * 1e-5
        target = target.cpu().numpy()
        for i_cls in range(n_cls):
            i_cls_idx = np.argwhere(target == i_cls).flatten()
            if len(i_cls_idx) > 0:
                cnts[i_cls] = len(i_cls_idx)
                accs[i_cls] = np.sum(correct[i_cls_idx])/len(i_cls_idx)*100

        return accs, cnts


def cls_accuracy_bc(output, target, cls=[0,1,2], delta=0.1):
    with torch.no_grad():
        accs = np.zeros([3, ])
        cnts = np.ones([3,])* 1e-7
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1)
        correct = pred.eq(target).cpu().numpy()
        for i in range(len(target)):
            if target[i] == cls[0]:
                accs[0] += correct[i]
                cnts[0] += 1
            elif target[i] == cls[1]:
                accs[1] += correct[i]
                cnts[1] += 1
            elif target[i] == cls[2]:
                i_correct = np.abs(output[i][0].cpu().numpy() - 0.5) < delta
                accs[2] += i_correct
                cnts[2] += 1
            else:
                raise ValueError(f'Out of range error! {target[i]} is given')
        accs = accs/ cnts *100
        return accs, cnts


def get_confusion_matrix_bc(output, target, cls=[-1,0,1], delta=0.1):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(-1).cpu().numpy()

        for i in range(len(target)):
            if target[i] == cls[0]:
                if np.abs(output[i][0].cpu().numpy()-0.5) < delta:
                    pred[i] = -1
                else:
                    continue

        pred = np.transpose(pred)
        cm = confusion_matrix(target.cpu().numpy(), pred)

        return cm, np.diag(cm)/np.sum(cm, axis=-1)


def get_confusion_matrix(output, target):
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        cm = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())

        return cm, np.diag(cm)/np.sum(cm, axis=-1)


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


def cross_entropy_loss_with_one_hot_labels(logits, labels):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(log_probs*labels, dim=1)
    return loss.mean()


def cross_entropy_loss_with_one_hot_labels_with_weights(logits, labels, weights):
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(log_probs*labels, dim=1) * weights
    return loss.mean()


def mix_ce_and_kl_loss(logits, labels, mask, alpha=1):
    inv_mask = mask.__invert__()
    log_probs = nn.functional.log_softmax(logits, dim=1)
    ce_loss = -torch.sum(log_probs[mask]*labels[mask], dim=1)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(log_probs[inv_mask], labels[inv_mask])
    loss = ce_loss.mean() + alpha*kl_loss
    return loss


def load_one_image(img_path):
    img = rasterio.open(img_path).read()[:3, :, :]
    return img


def load_one_image_cv2(img_path):
    img = cv2.imread(img_path)
    return img

def load_images(img_root, img_name_list, width=256, height=256):
    num_images = len(img_name_list)
    images = np.zeros([num_images, height, width, 3], dtype=np.uint8)
    for idx, img_path in enumerate(img_name_list):
        img = cv2.imread(os.path.join(img_root, img_path), cv2.IMREAD_COLOR)
        images[idx] = cv2.resize(img, (width, height))
    return images

def to_np(x):
    return x.cpu().detach().numpy()


def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now


def display_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'], param_group['initial_lr'])


def get_distribution(data):
    cls, cnt = np.unique(data, return_counts=True)
    for i_cls, i_cnt in zip(cls, cnt):
        print(f'{i_cls}: {i_cnt} ({i_cnt/len(data)*100:.2f}%)')
    print(f'total: {len(data)}')


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def log_configs(cfg, log_file='log.txt'):
    if os.path.exists(f'{cfg.save_folder}/{log_file}'):
        log_file = open(f'{cfg.save_folder}/{log_file}', 'a')
    else:
        log_file = open(f'{cfg.save_folder}/{log_file}', 'w')
    opt_dict = vars(cfg)
    for key in opt_dict.keys():
        write_log(log_file, f'{key}: {opt_dict[key]}')
    return log_file


def save_ckpt(cfg, model, optimizer, epoch, train_loss, angle_loss, dist_loss, center_loss, val_f1, postfix):
    # save all necessary information to resume training
    state = {
        'model': model.state_dict() if cfg.n_gpu <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'angle_loss': angle_loss,
        'dist_loss': dist_loss,
        'center_loss': center_loss,
        'val_f1': val_f1
    }
    save_file = os.path.join(cfg.save_folder, f'{postfix}')
    torch.save(state, save_file)
    print(f'ckpt saved to {save_file}.')


def set_wandb(cfg):
    wandb.login(key='##your key here##')
    wandb.init(project=cfg.project_name, tags=[cfg.dataset], name=cfg.experiment_name)
    wandb.config.update(cfg, allow_val_change=True)
    wandb.save('*.py')
    wandb.run.save()


def extract_embs(encoder, data_loader, return_labels=False, return_img_id = False, use_reduced_feature = True, vanilla_loader = False):
    encoder.eval()
    embs = []
    # inds = []
    if return_labels:
        labels = []
        ranks = []
    if return_img_id:
        img_ids = []
    # print(f'extracting embs')
    t1 = time.time()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if vanilla_loader:
                x_base, label, item = data
                rank = [-1] * len(label)
            else:
                x_base, label, rank, item = data

            print(f'{idx}/{len(data_loader)}', end='\r')
            x_base = x_base.float().cuda()
            if use_reduced_feature:
                embs.append(encoder(x_base).cpu())
            else:
                emm = encoder.avgpool(encoder.features(x_base)).squeeze().cpu()
                # normalize
                emm = emm / (emm.norm(dim=-1, keepdim=True) + 1e-7)
                embs.append(emm)
            # inds.append(item)
            if return_labels:
                labels.append(label)
                ranks.append(rank)
            if return_img_id:
                img_ids.append(item)
    # print(f'extracting embs done, time: {time.time()-t1:.2f}s')
    embs = torch.cat(embs)

    if return_labels:
        labels = np.concatenate(labels)
        ranks = np.concatenate(ranks)
        if return_img_id:
            img_ids = np.concatenate(img_ids)
            return embs, labels, ranks, img_ids
        else:
            return embs, labels, ranks
    else:
        return embs



def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x


def print_eval_result_by_groups_and_k(gt, ref_gt, preds_all, log_file, interval=10):
    test_cls_arr, cnt = np.unique(gt, return_counts=True)
    test_cls_min = test_cls_arr.min()
    test_cls_max = test_cls_arr.max()
    n_groups = int((test_cls_max - test_cls_min + 1) / interval + 0.5)

    title = 'Group \\ K |'
    for k in preds_all.keys():
        title += f" {k:<4} "
    title = title + ' | Best K | #Test | #Train '
    if log_file is not None:
        write_log(log_file, title)
    else:
        print(title)
    for i_group in range(n_groups):
        min_rank = interval * i_group
        max_rank = int(min(test_cls_max + 1, min_rank + interval))
        sample_idx_in_group = np.argwhere(np.logical_and(gt >= min_rank, gt < max_rank)).flatten()
        ref_sample_idx_in_group = np.argwhere(np.logical_and(ref_gt >= min_rank, ref_gt < max_rank)).flatten()

        if len(sample_idx_in_group) < 2:
            continue
        to_print = f' {min_rank:<3} ~ {max_rank - 1:<3} |'

        best_k_1 = -1
        best_mae = 1000
        for k in preds_all.keys():
            # ipdb.set_trace()
            i_group_errors_at_k = np.abs(preds_all[k][sample_idx_in_group] - gt[sample_idx_in_group])
            i_group_mean_at_k = np.mean(i_group_errors_at_k)
            to_print += f' {i_group_mean_at_k:.3f}' if i_group_mean_at_k<10 else f' {i_group_mean_at_k:.2f}'
            if i_group_mean_at_k < best_mae:
                best_mae = i_group_mean_at_k
                best_k_1 = k
        to_print += f' |   {best_k_1:<2}   | {len(sample_idx_in_group):<4}  | {len(ref_sample_idx_in_group):<4} '
        if log_file is not None:
            write_log(log_file, to_print)
        else:
            print(to_print)

    mean_all = '  Total   |'
    best_k_1 = -1
    best_mae = 1000
    for k in preds_all.keys():
        mean_at_k = np.mean(np.abs(preds_all[k] - gt))
        mean_all += f' {mean_at_k:.3f}'
        if mean_at_k < best_mae:
            best_mae = mean_at_k
            best_k_1 = k
    mean_all += f' |   {best_k_1:<2}   | {len(gt):<5} | {len(ref_gt):<5}'
    if log_file is not None:
        write_log(log_file, mean_all)
        write_log(log_file, f'Best Total MAE : {best_mae:.3f}\n')
    else:
        print(mean_all)
        print(f'Best Total MAE : {best_mae:.3f}\n')
    # r2 score
    if log_file is not None:
        write_log(log_file, title)
    else:
        print(title)
    interval2 = 100
    for i_group in range(n_groups):
        min_rank = interval2 * i_group
        max_rank = int(min(test_cls_max + 1, min_rank + interval2))
        sample_idx_in_group = np.argwhere(np.logical_and(gt >= min_rank, gt < max_rank)).flatten()
        ref_sample_idx_in_group = np.argwhere(np.logical_and(ref_gt >= min_rank, ref_gt < max_rank)).flatten()

        if len(sample_idx_in_group) < 2:
            continue
        to_print = f' {min_rank:<3} ~ {max_rank - 1:<3} |'
        best_k_2 = -1
        best_r2 = -1000
        for k in preds_all.keys():
            i_group_r2_at_k = r2_score(gt[sample_idx_in_group], preds_all[k][sample_idx_in_group])
            to_print += f' {i_group_r2_at_k:.3f}' if i_group_r2_at_k<10 else f' {i_group_r2_at_k:.2f}'
            if i_group_r2_at_k > best_r2:
                best_r2 = i_group_r2_at_k
                best_k_2 = k
        to_print += f' |   {best_k_2:<2}   | {len(sample_idx_in_group):<4}  | {len(ref_sample_idx_in_group):<4} '
        if log_file is not None:
            write_log(log_file, to_print)
        else:
            print(to_print)
    mean_all = '  Total   |'
    best_k_2 = -1
    best_r2 = -1000
    for k in preds_all.keys():
        r2_at_k = r2_score(gt, preds_all[k])
        mean_all += f' {r2_at_k:.3f}'
        if r2_at_k > best_r2:
            best_r2 = r2_at_k
            best_k_2 = k
    mean_all += f' |   {best_k_2:<2}   | {len(gt):<5} | {len(ref_gt):<5}'
    if log_file is not None:
        write_log(log_file, mean_all)
        write_log(log_file, f'Best Total R2 : {best_r2:.3f}\n')
    else:
        print(mean_all)
        print(f'Best Total R2 : {best_r2:.3f}\n')

    return best_mae, best_k_1, best_r2, best_k_2



def process_embs(embs, embs_train, train_labels, train_ranks, data_time, n_test, n_batch, cfg,
                 rec_k1=20, rec_k2=6,
                 rec_l=0.3
                 ):
    preds_all = defaultdict(list)
    preds_rank_all = defaultdict(list)
    with torch.no_grad():
        end = time.time()
        assert np.ndim(embs) == 2
        for idx in range(n_batch):
            print(f'processing batch: {idx}/{n_batch}', end='\r')
            data_time.update(time.time() - end)
            i_st = idx * cfg.batch_size_pred
            i_end = min(i_st + cfg.batch_size_pred, n_test)

            # ===================meters=====================
            if i_st < i_end:
                vals, inds = find_kNN(embs[i_st:i_end].view(i_end - i_st, -1), embs_train, k=max(cfg.k),
                                          metric=cfg.metric, reciprocal=cfg.reciprocal,
                                      rec_k1=rec_k1, rec_k2=rec_k2,
                                      rec_l = rec_l)

            else:
                continue

            inds = to_np(inds)
            for k in cfg.k: # tuning of k is now using the validation set
                nn_labels = train_labels[inds[:, :k]]

                if cfg.predict_scheme == 'mean':
                    pred_mean = np.mean(nn_labels, axis=-1, dtype=np.float32)


                elif cfg.predict_scheme == 'weighted-mean':
                    # predict using weightted mean, weight by distance, closer neighbors have higher weights
                    nn_dists = vals[:, :k].cpu().numpy() # retunred distances are negative
                    # normalize distances (updated Dec 19), solve zero division, add offset
                    nn_dists = -nn_dists / (np.sum(nn_dists, axis=-1, keepdims=True) + 1e-7)
                    # ipdb.set_trace()
                    nn_weights = np.exp(nn_dists)
                    # normalize weights
                    nn_weights = nn_weights / np.sum(nn_weights, axis=-1, keepdims=True)
                    pred_mean = np.sum(nn_labels * nn_weights, axis=-1)

                else:
                    raise ValueError(f'predict scheme {cfg.predict_scheme} is not supported')

                # if pred-mean contains nan
                if np.isnan(pred_mean).any():
                    ipdb.set_trace()
                    print('nan in pred_mean')

                preds_all[k].append(pred_mean)

                nn_ranks = train_ranks[inds[:, :k]]
                pred_rank = []
                # majority vote to get rank
                for j in range(len(nn_ranks)):
                    cur = np.bincount(nn_ranks[j], minlength=cfg.ref_point_num)
                    # ipdb.set_trace() # check entropy of nn ranks
                    if np.sum(cur == np.max(cur)) > 1:
                        pred_rank.append(np.round(nn_ranks[j].mean()))

                    else:
                        pred_rank.append(np.argmax(cur))

                    if k == cfg.k[-1]:
                        # save mean std
                        pred_all_rank_std = np.std(nn_ranks, axis=-1).mean()

                preds_rank_all[k].append(pred_rank)


    for key in preds_all.keys():
        preds_all[key] = np.concatenate(preds_all[key])
        preds_rank_all[key] = np.concatenate(preds_rank_all[key])

    return preds_all, preds_rank_all, pred_all_rank_std


def process_embs_fixed_k_search(embs, embs_train, train_labels, train_ranks, data_time, n_test, n_batch, cfg,
                 rec_k1=20, rec_k2=6,
                 rec_l=0.3):

    '''fixing k for knn search'''
    preds_all = []
    preds_rank_all = []
    with torch.no_grad():
        end = time.time()
        assert np.ndim(embs) == 2
        for idx in range(n_batch):

            data_time.update(time.time() - end)
            i_st = idx * cfg.batch_size_pred
            i_end = min(i_st + cfg.batch_size_pred, n_test)

            vals, inds = find_kNN(embs[i_st:i_end].view(i_end - i_st, -1), embs_train, k=cfg.init_knn_k,
                                      metric=cfg.metric, reciprocal=cfg.reciprocal,
                                  rec_k1=rec_k1, rec_k2=rec_k2,
                                  rec_l = rec_l)

            inds = to_np(inds)
            nn_labels = train_labels[inds[:, :cfg.init_knn_k]]

            if cfg.predict_scheme == 'mean':
                pred_mean = np.mean(nn_labels, axis=-1, dtype=np.float32)


            elif cfg.predict_scheme == 'weighted-mean':
                # predict using weightted mean, weight by distance, closer neighbors have higher weights
                nn_dists = vals[:, :cfg.init_knn_k].cpu().numpy() # retunred distances are negative
                nn_weights = np.exp(nn_dists)
                # normalize weights
                nn_weights = nn_weights / np.sum(nn_weights, axis=-1, keepdims=True)
                pred_mean = np.sum(nn_labels * nn_weights, axis=-1)

            else:
                raise ValueError(f'predict scheme {cfg.predict_scheme} is not supported')

            preds_all.extend(pred_mean)

            nn_ranks = train_ranks[inds[:, :cfg.init_knn_k]]
            pred_rank = []
            # majority vote to get rank
            for j in range(len(nn_ranks)):
                cur = np.bincount(nn_ranks[j], minlength=cfg.ref_point_num)
                if np.sum(cur == np.max(cur)) > 1:
                    pred_rank.append(np.round(nn_ranks[j].mean()))

                else:
                    pred_rank.append(np.argmax(cur))

            preds_rank_all.extend(pred_rank)

    preds_all = np.array(preds_all)
    preds_rank_all = np.array(preds_rank_all)

    return preds_all, preds_rank_all


def prepare_few_shot_data(embs_test, test_labels, test_ranks, cfg):
    test_ref_ind = []
    for rank in range(cfg.n_ranks):
        idx = np.where(test_ranks == rank)[0]
        idx = np.random.choice(idx, cfg.few_shot_num, replace=True)
        test_ref_ind.append(idx)


    test_ref_ind = np.concatenate(test_ref_ind)
    embs_test_ref = embs_test[test_ref_ind]
    test_labels_ref = test_labels[test_ref_ind]
    test_ranks_ref = test_ranks[test_ref_ind]

    # the rest of test data as the new test data
    test_test_ind = np.setdiff1d(np.arange(len(embs_test)), test_ref_ind)
    embs_test = embs_test[test_test_ind]
    test_labels = test_labels[test_test_ind]
    test_ranks = test_ranks[test_test_ind]
    return embs_test, test_labels, test_ranks, embs_test_ref, test_labels_ref, test_ranks_ref, test_ref_ind


def prepare_few_shot_data_regressor(preds, labels, ranks, cfg):
    test_ref_ind = []
    for rank in range(5):
        idx = np.where(ranks == rank)[0]
        idx = np.random.choice(idx, cfg.few_shot_num, replace=True)
        test_ref_ind.append(idx)

    test_ref_ind = np.concatenate(test_ref_ind)
    preds_ref = preds[test_ref_ind]
    labels_ref = labels[test_ref_ind]
    ranks_ref = ranks[test_ref_ind]

    # the rest of test data as the new test data
    test_test_ind = np.setdiff1d(np.arange(len(preds)), test_ref_ind)
    preds_test = preds[test_test_ind]
    labels_test = labels[test_test_ind]
    ranks_test = ranks[test_test_ind]
    return preds_test, labels_test, ranks_test, preds_ref, labels_ref, ranks_ref, test_ref_ind


def query_knn_head(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref, test_labels, test_ranks, cfg, lim):



    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))
    data_time = AverageMeter()

    preds_all, preds_rank_all = process_embs_fixed_k_search(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref, data_time,
                                                            n_test, n_batch, cfg,
                                                            rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                            rec_l=cfg.reciprocal_l)


    if cfg.logscale:
        preds_all = np.exp(preds_all) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds_all))
    rmse = np.sqrt(np.mean((test_labels - preds_all) ** 2))
    r2 = r2_score(test_labels, preds_all)

    f1 = f1_score(test_ranks, preds_rank_all, average='micro')

    return mae, rmse, r2, f1


def query_prototypical_head(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref, test_labels, test_ranks, cfg):
    # build prototype for each class by taking the mean of embeddings belonging to that class
    # then predict by finding the nearest prototype
    embs_test = embs_test.cpu()
    # embs_test_ref = embs_test_ref.cpu().numpy()

    prototypes = torch.zeros([cfg.n_ranks, embs_test_ref.shape[1]])
    prototype_labels = []
    prototype_ranks = []
    for i_cls in range(cfg.n_ranks):
        # mean prototype
        if cfg.proto_build == 'mean':
            prototypes[i_cls] = torch.mean(embs_test_ref[test_ranks_ref == i_cls], dim=0)

        elif cfg.proto_build == 'median':
            prototypes[i_cls] = torch.median(embs_test_ref[test_ranks_ref == i_cls], dim=0)[0]

        elif cfg.proto_build == 'kmean':
            # kmean prototype
            from sklearn.cluster import KMeans
            # ipdb.set_trace()
            embs_test_ref = embs_test_ref.cpu()
            kmeans = KMeans(n_clusters=1, random_state=0).fit(embs_test_ref[test_ranks_ref == i_cls])
            prototypes[i_cls] = torch.from_numpy(kmeans.cluster_centers_[0])
        else:
            raise ValueError(f'prototype build method {cfg.proto_build} is not supported')
        prototype_labels.append(np.mean(test_labels_ref[test_ranks_ref == i_cls], axis=0))
        prototype_ranks.append(i_cls)


    # average Euclidean distance between prototypes
    dis = 0
    for i in range(len(prototypes)):
        for j in range(i+1, len(prototypes)):
            dis += torch.norm(prototypes[i] - prototypes[j])
    dis = dis / (len(prototypes) * (len(prototypes) - 1) / 2)
    # print('average Euclidean distance between prototypes: ', dis)

    # average cosine similarity between prototypes
    sim = 0
    for i in range(len(prototypes)):
        for j in range(i+1, len(prototypes)):
            sim += torch.cosine_similarity(prototypes[i], prototypes[j], dim=0)
    sim = sim / (len(prototypes) * (len(prototypes) - 1) / 2)
    # print('average cosine similarity between prototypes: ', sim)

    prototype_labels = np.stack(prototype_labels, axis=0)
    prototype_ranks = np.stack(prototype_ranks, axis=0)
    # print('shape of prototypes: ', prototypes.shape)
    # print('shape of prototype_labels: ', prototype_labels.shape)
    # print('prototype labels: ', prototype_labels)
    # ipdb.set_trace()
    preds = []
    pred_ranks = []
    proto_ids = []
    for i in range(len(embs_test)):

        val, ind = find_kNN(embs_test[i], prototypes, k=1, metric='cosine', reciprocal=False, rec_k1=20, rec_k2=6, rec_l=0.3)
        pred = prototype_labels[ind]
        preds.append(pred)
        pred_rank = prototype_ranks[ind]
        pred_ranks.append(pred_rank)
        proto_ids.append(ind)

    preds = np.array(preds)
    pred_ranks = np.array(pred_ranks)
    proto_ids = np.array(proto_ids)
    # ipdb.set_trace()
    # r2 and mae
    if cfg.logscale:
        preds = np.exp(preds) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds))
    rmae = mae / np.mean(test_labels)
    r2 = r2_score(test_labels, preds)

    # pred rank
    f1 = f1_score(test_ranks, pred_ranks, average='micro')
    # print('pred rank range: ', np.min(pred_ranks), np.max(pred_ranks))
    # print('target rank range: ', np.min(test_ranks), np.max(test_ranks))


    return mae, rmae, r2, f1, proto_ids, prototypes, prototype_labels, pred_ranks



def query_prototypical_dist_wei_head(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref, test_labels, test_ranks, cfg):
    # build prototype for each class by taking the mean of embeddings belonging to that class
    # then predict by finding the nearest prototype
    embs_test = embs_test.cpu()
    embs_test_ref = embs_test_ref.cpu()

    prototypes = torch.zeros([cfg.n_ranks, embs_test_ref.shape[1]])
    prototype_ranks = []
    for i_cls in range(cfg.n_ranks):
        # mean prototype
        if cfg.proto_build == 'mean':
            prototypes[i_cls] = torch.mean(embs_test_ref[test_ranks_ref == i_cls], dim=0)

        elif cfg.proto_build == 'median':
            prototypes[i_cls] = torch.median(embs_test_ref[test_ranks_ref == i_cls], dim=0)[0]

        elif cfg.proto_build == 'kmean':
            # kmean prototype
            from sklearn.cluster import KMeans
            # ipdb.set_trace()
            embs_test_ref = embs_test_ref.cpu()
            kmeans = KMeans(n_clusters=1, random_state=0).fit(embs_test_ref[test_ranks_ref == i_cls])
            prototypes[i_cls] = torch.from_numpy(kmeans.cluster_centers_[0])
        else:
            raise ValueError(f'prototype build method {cfg.proto_build} is not supported')
        prototype_ranks.append(i_cls)


    prototype_ranks = np.stack(prototype_ranks, axis=0)

    preds = []
    pred_ranks = []
    for i in range(len(embs_test)):

        val, ind = find_kNN(embs_test[i], prototypes, k=1, metric='cosine', reciprocal=False, rec_k1=20, rec_k2=6, rec_l=0.3)

        # all embeddings of the selected prototype
        embs_proto = embs_test_ref[test_ranks_ref == ind.item()]
        # and embeddings of the selected prototype
        # ipdb.set_trace()
        embs_proto = torch.cat([embs_proto, prototypes[ind]], dim=0)

        # all labels of the selected prototype
        labels_proto = test_labels_ref[test_ranks_ref == ind.item()]
        # add prototype mean label as well!!!
        lb_proto_mean = np.mean(labels_proto)
        labels_proto = np.concatenate([labels_proto, [lb_proto_mean]], axis=0)

        # distance between query and all embeddings of the selected prototype
        dists = torch.norm(embs_proto - embs_test[i], dim=1)
        # weight by distance, closer neighbors have higher weights
        weights = 1 / (dists + 1e-7)
        # normalize weights
        weights = weights / torch.sum(weights)
        weights = np.array(weights)
        # predict using weightted mean
        # ipdb.set_trace()
        pred = np.sum(labels_proto * weights, axis=0)


        preds.append(pred)
        pred_rank = prototype_ranks[ind]
        pred_ranks.append(pred_rank)

    preds = np.array(preds)
    pred_ranks = np.array(pred_ranks)
    # ipdb.set_trace()
    # r2 and mae
    if cfg.logscale:
        preds = np.exp(preds) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds))
    rmae = mae / np.mean(test_labels)
    r2 = r2_score(test_labels, preds)

    # pred rank
    f1 = f1_score(test_ranks, pred_ranks, average='micro')
    # print('pred rank range: ', np.min(pred_ranks), np.max(pred_ranks))
    # print('target rank range: ', np.min(test_ranks), np.max(test_ranks))


    return mae, rmae, r2, f1

def calculate_dis(train_ranks, test_ranks, embs_train, embs_test, cfg):
    # calculate distance between corresponding protypes in train and test
    right_dis = []
    right_sim = []
    all_other_dis = []
    all_other_sim = []
    adj_dis = []
    adj_sim = []
    for r in range(cfg.class_number):
        # ipdb.set_trace()
        print('============================')
        idx_train = np.where(train_ranks == r)[0]
        idx_test = np.where(test_ranks == r)[0]
        print('rank: ', r)
        proto_train = torch.mean(embs_train[idx_train], dim=0)
        proto_test = torch.mean(embs_test[idx_test], dim=0)
        print('shape of proto: ', proto_train.shape)
        # euclidean distance between the two
        dist = torch.norm(proto_train - proto_test).cpu().detach().numpy()
        # cosine similarity
        cos_sim = torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy()
        print('euclidean distance: ', dist)
        print('cosine similarity: ', cos_sim)
        right_dis.append(dist)
        right_sim.append(cos_sim)

        # average distance between the current test rank to other ref train protos
        dists = []
        sims = []
        for i in range(cfg.class_number):
            if i != r:
                idx_train = np.where(train_ranks == i)[0]
                proto_train = torch.mean(embs_train[idx_train], dim=0)
                dists.append(torch.norm(proto_train - proto_test).cpu().detach().numpy())
                sims.append(torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy())
        print('average distance to all other ref protos: ', np.mean(np.array(dists)))
        print('average cosine similarity to all other ref protos: ', np.mean(np.array(sims)))
        all_other_dis.append(np.mean(np.array(dists)))
        all_other_sim.append(np.mean(np.array(sims)))

        # average distance between current test rank to adjacent ranks in ref
        if r == 0:
            idx_train = np.where(train_ranks == r + 1)[0]
            proto_train = torch.mean(embs_train[idx_train], dim=0)
            dist = torch.norm(proto_train - proto_test).cpu().detach().numpy()
            sim = torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy()
            print('average distance to adjacent ref protos: ', dist)
            print('average cosine similarity to adjacent ref protos: ', sim)
            adj_dis.append(dist)
            adj_sim.append(sim)
        elif r == cfg.class_number - 1:
            idx_train = np.where(train_ranks == r - 1)[0]
            proto_train = torch.mean(embs_train[idx_train], dim=0)
            dist = torch.norm(proto_train - proto_test).cpu().detach().numpy()
            sim = torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy()
            print('average distance to adjacent ref protos: ', dist)
            print('average cosine similarity to adjacent ref protos: ', sim)
            adj_dis.append(dist)
            adj_sim.append(sim)
        else:
            dists = []
            sims = []
            idx_train = np.where(train_ranks == r - 1)[0]
            proto_train = torch.mean(embs_train[idx_train], dim=0)
            dists.append(torch.norm(proto_train - proto_test).cpu().detach().numpy())
            sims.append(torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy())
            idx_train = np.where(train_ranks == r + 1)[0]
            proto_train = torch.mean(embs_train[idx_train], dim=0)
            dists.append(torch.norm(proto_train - proto_test).cpu().detach().numpy())
            sims.append(torch.cosine_similarity(proto_train, proto_test, dim=0).cpu().detach().numpy())
            print(sims)
            print('average distance to adjacent ref protos: ', np.mean(np.array(dists)))
            print('average cosine similarity to adjacent ref protos: ', np.mean(np.array(sims)))

            adj_dis.append(np.mean(np.array(dists)))
            adj_sim.append(np.mean(np.array(sims)))

    return right_dis, right_sim, all_other_dis, all_other_sim, adj_dis, adj_sim


def query_regressor_linear_calib(preds_test, labels_test, preds_ref, labels_ref):
    from sklearn.linear_model import LinearRegression
    head = LinearRegression(fit_intercept=True)
    head.fit(preds_ref.reshape(-1, 1), labels_ref)
    calib_preds = head.predict(preds_test.reshape(-1, 1))
    r2 = r2_score(labels_test, calib_preds)
    # rmse
    rmse = np.sqrt(np.mean((labels_test - calib_preds) ** 2))
    return r2, rmse





def query_linear_fitting(embs_test, embs_test_ref, test_labels_ref, test_labels, cfg, lim):
    # linear regression
    from sklearn.linear_model import LinearRegression

    # ipdb.set_trace()
    embs_test = embs_test.cpu().numpy()
    embs_test_ref = embs_test_ref.cpu().numpy()
    if cfg.model == 'regressor':
        # normalize # for regressor, model does not have a normalization layer
        # embs_test = (embs_test - embs_test.mean(axis=0)) / embs_test.std(axis=0)
        # embs_test_ref = (embs_test_ref - embs_test_ref.mean(axis=0)) / embs_test_ref.std(axis=0)
        embs_test = embs_test / np.linalg.norm(embs_test, axis=1, keepdims=True)
        embs_test_ref = embs_test_ref / np.linalg.norm(embs_test_ref, axis=1, keepdims=True)


    if cfg.reduce_dim_one:
        print('reducing embedding dimension to 1 using tsne')
        # reduce embedding dimension to 1 using tsne and performance regression
        from sklearn.manifold import TSNE
        tsne = TSNE(init = 'pca', n_components=1, verbose=0, perplexity=40, n_iter=300)
        # reduce fit on both ref and test data
        embs_test_rd = tsne.fit_transform(np.concatenate([embs_test, embs_test_ref]))
        embs_test = embs_test_rd[:len(embs_test)]
        embs_test_ref = embs_test_rd[len(embs_test):]

    head = LinearRegression()
    head.fit(embs_test_ref, test_labels_ref)

    preds = head.predict(embs_test)

    if cfg.logscale:
        preds = np.exp(preds) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds))
    rmae = mae / np.mean(test_labels)
    r2 = r2_score(test_labels, preds)

    return mae, rmae, r2, preds



def query_diffusion_head(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref, test_labels, test_rank,
                         cfg, lim, return_preds = False, return_diffuser = False):

    # diffuse to get predictions, using fixed k in knn search
    diffuser = Diffuser(cfg)
    embs_X = np.concatenate([embs_test_ref.cpu().numpy(), embs_test.cpu().numpy()], axis=0)
    diffuser.diffuse_pred(embs_X, test_ranks_ref, test_labels_ref)
    preds_rank = diffuser.p_labels[len(test_ranks_ref):]
    preds_all = diffuser.p_values
    prob_values = diffuser.probs_value
    extract_probs = diffuser.extract_probs # sorted already!

    # map from log space to linear space
    if cfg.logscale:
        preds_all = np.exp(preds_all) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset
        test_labels_ref = np.exp(test_labels_ref) - cfg.offset

    if cfg.diffuse_calibrate:
        # calibrate using support data
        preds_ref = preds_all[:len(test_ranks_ref)]
        lr = LinearRegression()
        lr.fit(preds_ref.reshape(-1, 1), test_labels_ref)
        preds_final = lr.predict(preds_all[len(test_ranks_ref):].reshape(-1, 1)).squeeze()

    else:
        preds_final = preds_all[len(test_ranks_ref):]


    def draw_once(sd):
        # split test data into 5 intervals
        select_for_vis = []
        for i in range(5):
            idx = np.where(test_rank == i)[0]
            np.random.seed(sd)
            idx = np.random.choice(idx, 10, replace=False)
            select_for_vis.append(idx)
        select_for_vis = np.concatenate(select_for_vis)
        select_for_vis_labels = test_labels[select_for_vis]
        select_for_vis_preds = preds_final[select_for_vis]
        select_for_vis_probs = prob_values[25:][select_for_vis]
        # get the 5th largest value in each row
        thres = np.sort(select_for_vis_probs, axis=1)[:, -3]
        # for each row, set those below the threshold to 0
        select_for_vis_probs[select_for_vis_probs < thres[:, None]] = 0

        # normalize color rowwise
        select_for_vis_probs_nm = (select_for_vis_probs - np.min(select_for_vis_probs, axis=1, keepdims=True)) / (np.max(select_for_vis_probs, axis=1, keepdims=True) - np.min(select_for_vis_probs, axis=1, keepdims=True))

        subfig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # plot first
        ax[0].matshow(select_for_vis_probs_nm, cmap='summer')

        # convert test_labels to integer
        test_labels_int = np.round(test_labels[select_for_vis]).astype(int)
        # make onehot
        test_labels_onehot = np.zeros((len(test_labels_int), max(test_labels_int) + 1))
        test_labels_onehot[np.arange(len(test_labels_int)), test_labels_int] = 1
        # plt.matshow(test_labels_onehot, cmap = 'summer')
        ax[1].matshow(test_labels_onehot, cmap='summer')

        # convert select_pred also to onehot
        select_for_vis_preds_int = np.round(select_for_vis_preds).astype(int)
        select_for_vis_preds_onehot = np.zeros((len(select_for_vis_preds_int), max(test_labels_int) + 1))
        select_for_vis_preds_onehot[np.arange(len(select_for_vis_preds_int)), select_for_vis_preds_int] = 1
        # plt.matshow(select_for_vis_preds_onehot, cmap = 'summer')
        ax[2].matshow(select_for_vis_preds_onehot, cmap='summer')
        # no ticks
        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()

    # draw_once(10)
    # s0 = np.zeros((75, 25))
    # for j in range(25):
    #     s0[j, j] = 1
    # plt.matshow(s0, cmap='summer')
    # # ticks off
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()







    ipdb.set_trace()






    mae = np.mean(np.abs(test_labels - preds_final))
    rmse = np.sqrt(np.mean((test_labels - preds_final) ** 2))
    r2 = r2_score(test_labels, preds_final)
    f1 = f1_score(test_rank, preds_rank, average='micro')

    if return_preds:
        if return_diffuser:
            return mae, rmse, r2, f1, preds_final, test_labels, diffuser
        else:
            return mae, rmse, r2, f1, preds_final, test_labels
    else:
        if return_diffuser:
            return mae, rmse, r2, f1, diffuser
        else:
            return mae, rmse, r2, f1



def plot_diffusion(embs, query_idx, indx, label_values_all, extract_probs, tsne = True, font = 16, cmap = 'winter'):
    """labeled and unlabeled concatenated, all equal to length of total"""
    # normalize label values
    label_values_all = (label_values_all - np.min(label_values_all)) / (np.max(label_values_all) - np.min(label_values_all))
    if tsne:
        reducer = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500)
    else:
        reducer = PCA(n_components=2)
    embs_reduced = reducer.fit_transform(embs)
    # color points based differences in the labels
    distances = abs(label_values_all[query_idx] - label_values_all[:25]) # distance to all known labels (in log scale)
    # map to color
    colors = distances / np.max(distances)
    fig = plt.subplots(1, 2, figsize=(10, 5))
    # plot colormap
    ax = plt.subplot(1, 2, 1)
    ax.scatter(embs_reduced[:, 0], embs_reduced[:, 1], c=label_values_all, cmap=cmap, s=2, label='All data', alpha = 0.5, vmin=0, vmax=1)

    ax.scatter(embs_reduced[:25, 0], embs_reduced[:25, 1], c=label_values_all[:25], cmap=cmap, s=180, label='Support', vmin=0, vmax=1)
    # normalize color to get color for query
    cc = label_values_all[query_idx]
    ax.scatter(embs_reduced[query_idx, 0], embs_reduced[query_idx, 1], c=cc, cmap = cmap, s=450, marker='X', label = 'Query', vmin=0, vmax=1, edgecolors='gray')
    # make size of marker in indx proportional to the value in extract_probs
    sizes = extract_probs[query_idx] * 1000
    ax.scatter(embs_reduced[indx[query_idx], 0], embs_reduced[indx[query_idx], 1], c='r', s=sizes, marker='*', label = 'Diffusion activated')


    plt.xticks([])
    plt.yticks([])

    ax.set_title('GOL+MDR', fontsize = font)

    ax2 = plt.subplot(1, 2, 2)
    # calculate indx based on knn
    # KNN search in embs[:25]
    # convert to tensor
    embs_tf = torch.from_numpy(embs)
    vals, indx_knn = find_kNN(embs_tf, embs_tf[:25], k=10,
                          metric='L2', reciprocal=False,
                          rec_k1=0, rec_k2=0,
                          rec_l=0)
    ax2.scatter(embs_reduced[:, 0], embs_reduced[:, 1], c=label_values_all, cmap=cmap, s=2, label='all data', alpha = 0.5, vmin=0, vmax=1)

    ax2.scatter(embs_reduced[:25, 0], embs_reduced[:25, 1], c=label_values_all[:25], cmap=cmap, s=180, label='Support', vmin=0, vmax=1)
    ax2.scatter(embs_reduced[query_idx, 0], embs_reduced[query_idx, 1], c=label_values_all[query_idx], cmap=cmap, s=450, marker='X', label = 'Query', vmin=0, vmax=1, edgecolors='gray')
    # size proportional to the simialrity
    knn_dists = vals[:, :10].cpu().numpy()
    # ipdb.set_trace()
    nn_weights = np.exp(knn_dists)
    # normalize weights
    nn_weights = nn_weights / np.sum(nn_weights, axis=-1, keepdims=True)
    # exp scale
    sizes = nn_weights[query_idx] * 1200
    ax2.scatter(embs_reduced[indx_knn[query_idx], 0], embs_reduced[indx_knn[query_idx], 1], c='r', s=sizes, marker='*', label = 'kNN activated')

    plt.xticks([])
    plt.yticks([])
    ax2.set_title('GOL', fontsize = font)


    plt.show()


def query_prototype_diffusion_head(embs_test, embs_test_ref, test_labels_ref, test_ranks_ref,
                                   test_labels, test_rank, proto_ids, embs_proto, label_proto, pred_proto_ranks, cfg,
                                   lim):
    """update ref embedding by moving those closer to embedding of prototypes"""

    # diffuse to get predictions, using prototype as reference
    embs_test_ref = embs_test_ref.cpu().numpy()
    embs_test = embs_test.cpu().numpy()
    embs_proto = embs_proto.cpu().numpy()


    # update ref embedding by moving those closer to embedding of prototypes
    embs_test_ref_proto = embs_test_ref.copy()
    for i in range(len(embs_test_ref)):
        # ipdb.set_trace()
        proto_emb = embs_proto[test_ranks_ref[i]]
        # move ref embedding closer to prototype by taking weighted average
        # ipdb.set_trace()
        embs_test_ref_proto[i] = cfg.diffuse_proto_eta * embs_test_ref[i] + (1 - cfg.diffuse_proto_eta) * proto_emb
        # normalize
        embs_test_ref_proto[i] = embs_test_ref_proto[i] / np.linalg.norm(embs_test_ref_proto[i])

    embs_X = np.concatenate([embs_test_ref_proto, embs_test], axis=0)
    diffuser = Diffuser(cfg)
    diffuser.diffuse_pred(embs_X, test_ranks_ref, test_labels_ref)

    preds_rank = diffuser.p_labels[len(test_ranks_ref):]
    preds = diffuser.p_values[len(test_ranks_ref):]

    if cfg.logscale:
        preds = np.exp(preds) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds))
    rmae = mae / np.mean(test_labels)
    r2 = r2_score(test_labels, preds)
    # plot_scatter(test_labels, preds, 'Predicted vs. target', 'Reference',
    #                 'Predicted'
    #                 , lim, font=35, spinexy=0, markersize=30, xtic=1, showr2=1, showfit=0)
    # ipdb.set_trace()
    return mae, rmae, r2, None


def query_logistic_fitting(embs_test, embs_test_ref, labels_ref, labels_test, ranks_ref, ranks_test, cfg, lim):
    # logistic regression
    from sklearn.linear_model import LogisticRegression
    embs_test = embs_test.cpu().numpy()
    embs_test_ref = embs_test_ref.cpu().numpy()
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(embs_test_ref, ranks_ref)
    preds_rank = clf.predict(embs_test)
    preds = np.zeros_like(preds_rank)
    for r in range(cfg.n_ranks):
        preds[preds_rank == r] = np.mean(labels_ref[ranks_ref == r])

    mae = np.mean(np.abs(labels_test - preds))
    rmae = mae / np.mean(labels_test)
    r2 = r2_score(labels_test, preds)
    f1 = f1_score(ranks_test, preds_rank, average='micro')
    return mae, rmae, r2, f1


def query_spectral_projection_head(embs_test, embs_test_ref, test_labels_ref, test_labels, cfg):
    X = np.concatenate([embs_test_ref.cpu().numpy(), embs_test.cpu().numpy()], axis=0)
    X = X / np.linalg.norm(X, axis=1)[:, None]
    D = np.dot(X, X.T)
    # N = X.shape[0]
    # I = np.argsort(-D, axis=1)
    # # Create the graph
    # I = I[:, 1:]
    # W = np.zeros((N, N))
    #
    # for i in range(N):
    #     # using fixed k
    #     W[i, I[i, :cfg.diffuse_k]] = D[i, I[i, :cfg.diffuse_k]] ** cfg.diffuse_gamma
    #         # ipdb.set_trace()
    # # ipdb.set_trace()
    #
    # W = W + W.T
    #
    # # Normalize the graph
    # W = W - np.diag(np.diag(W))
    # S = W.sum(axis=1)
    # S[S == 0] = 1
    # D = np.array(1. / np.sqrt(S))
    # Wn = D * W * D


    sc = SpectralEmbedding(n_components=1, affinity='precomputed',
                           n_neighbors=cfg.spectral_emb_k, n_jobs=cfg.spectral_emb_njobs)
    # ipdb.set_trace()
    maps = sc.fit_transform(D)

    lr = LinearRegression()
    lr.fit(maps[:len(embs_test_ref)], test_labels_ref)
    preds = lr.predict(maps[len(embs_test_ref):])
    # ipdb.set_trace()


    if cfg.logscale:
        preds = np.exp(preds) - cfg.offset
        test_labels = np.exp(test_labels) - cfg.offset

    mae = np.mean(np.abs(test_labels - preds))
    rmae = mae / np.mean(test_labels)
    r2 = r2_score(test_labels, preds)
    return mae, rmae, r2, preds










def log_alpha_preds(preds_all, preds_all_alpha, gt, bestk1, bestk2, bestk1_a, bestk2_a, epoch):
    """
    scores based on different embedding processing

    """
    def get_r2(predall, k, gt):
        preds = predall[k]
        return r2_score(gt, preds)

    log_names= ['vanilla mae', 'vanilla r2', 'alpha mae', 'alpha r2']
    r2_vanilla_mae = get_r2(preds_all, bestk1, gt)
    r2_vanilla_r2 = get_r2(preds_all, bestk2, gt)
    r2_alpha_mae = get_r2(preds_all_alpha, bestk1_a, gt)
    r2_alpha_r2 = get_r2(preds_all_alpha, bestk2_a, gt)

    # wandb bar plot
    data = [[log_names[0], r2_vanilla_mae], [log_names[1], r2_vanilla_r2], [log_names[2], r2_alpha_mae], [log_names[3], r2_alpha_r2]]
    table = wandb.Table(data=data, columns=['scheme', 'r2_score'])
    wandb.log({"r2_scores": wandb.plot.bar(table, "scheme", "r2_score",
                                            title="R2 scores")}, step=epoch)
    return






def log_eval_preds(preds_all, gt, bestk1, bestk2, epoch):
    # preds based on best k1, mae
    log_names= ['mae_k', 'r2_k']
    preds1 = preds_all[bestk1]

    agb_data = [[label, value] for (label, value) in zip(gt, np.array(preds1))]
    agb_table = wandb.Table(data=agb_data, columns=['agb_label', 'pred'])
    wandb.log({"validset_agb_prediction_agb bestmodel scheme = " + log_names[0]:
                   wandb.plot.scatter(agb_table, "agb_label", "pred",
                                      title="AGB estimation scatter plot - Model scheme: " + log_names[0])}, step=epoch)

    preds2 = preds_all[bestk2]
    agb_data = [[label, value] for (label, value) in zip(gt, np.array(preds2))]
    agb_table = wandb.Table(data=agb_data, columns=['agb_label', 'pred'])
    wandb.log({"validset_agb_prediction_agb bestmodel scheme = " + log_names[1]:
                   wandb.plot.scatter(agb_table, "agb_label", "pred",
                                      title="AGB estimation scatter plot - Model scheme: " + log_names[1])}, step=epoch)

    return

def log_eval_preds_vanilla(preds_all, gt, epoch, set = 'validset'):
    data = [[label, value] for (label, value) in zip(gt, np.array(preds_all))]
    table = wandb.Table(data=data, columns=['label', 'pred'])
    wandb.log({set + "_prediction":
                     wandb.plot.scatter(table, "label", "pred",
                                          title=set+" scatter plot")}, step=epoch)

    return




def log_eval_preds_vanila(preds_all, gt, epoch):
    # preds based on best k1, mae


    agb_data = [[label, value] for (label, value) in zip(gt, np.array(preds_all))]
    agb_table = wandb.Table(data=agb_data, columns=['agb_label', 'pred'])
    wandb.log({"validset_agb_prediction_agb bestmodel":
                   wandb.plot.scatter(agb_table, "agb_label", "pred",
                                      title="AGB estimation scatter plot")}, step=epoch)

    return


def log_eval_ranks(preds_rank_all, gt_rank, bestk1, bestk2, epoch):
    """
    log predicted ranks and gt ranks
    """
    log_names= ['mae_k', 'r2_k']
    preds1 = preds_rank_all[bestk1]
    unique_ranks = np.arange(0, 10) # 10 ranks


    # 2d histogram of predicted ranks and gt ranks
    data = [[label, value] for (label, value) in zip(gt_rank, np.array(preds1))]
    table = wandb.Table(data=data, columns=['gt_rank', 'pred_rank'])
    wandb.log({"validset_agb_prediction_rank bestmodel scheme = " + log_names[0]:
            wandb.plot.confusion_matrix(probs=None,
                                        y_true=gt_rank,
                                        preds=preds1,
                                        class_names=unique_ranks)}, step=epoch)

    preds2 = preds_rank_all[bestk2]
    data = [[label, value] for (label, value) in zip(gt_rank, np.array(preds2))]
    table = wandb.Table(data=data, columns=['gt_rank', 'pred_rank'])
    wandb.log({"validset_agb_prediction_rank bestmodel scheme = " + log_names[1]:
            wandb.plot.confusion_matrix(probs=None,
                                        y_true=gt_rank,
                                        preds=preds2,
                                        class_names=unique_ranks)}, step=epoch)


    # f1 scores
    f1_score_mac = f1_score(gt_rank, preds1, average='macro')
    f1_score_mic = f1_score(gt_rank, preds1, average='micro')


    return f1_score_mac, f1_score_mic


def log_eval_ranks_vanilla(preds_rank_all, gt_rank, epoch, set = 'validset'):
    """
    log predicted ranks and gt ranks
    """
    unique_ranks = np.arange(0, 5)

    # 2d histogram of predicted ranks and gt ranks
    data = [[label, value] for (label, value) in zip(gt_rank, np.array(preds_rank_all))]
    table = wandb.Table(data=data, columns=['gt_rank', 'pred_rank'])
    wandb.log({set+"_prediction_rank":
            wandb.plot.confusion_matrix(probs=None,
                                        y_true=gt_rank,
                                        preds=preds_rank_all,
                                        class_names=unique_ranks)}, step=epoch)

    # f1 scores
    f1_score_mac = f1_score(gt_rank, preds_rank_all, average='macro')
    f1_score_mic = f1_score(gt_rank, preds_rank_all, average='micro')

    return f1_score_mac, f1_score_mic


def log_embeddings(embs, labels, epoch, ranks,
                   stage = 'train',
                   colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black'],
                   colors2 = ['rosybrown', 'red', 'orange', 'yellow', 'lawngreen', 'green', 'cyan', 'blue', 'violet', 'purple', 'rosybrown', 'red', 'orange', 'yellow', 'lawngreen', 'green', 'cyan', 'blue', 'violet', 'purple']):


    tsne = TSNE(init = 'pca', n_components=2, verbose=0, perplexity=40, n_iter=300)
    embs_train_reduced = tsne.fit_transform(embs.cpu())

    tsne1 = TSNE(init = 'pca', n_components=1, verbose=0, perplexity=40, n_iter=300)
    embs_train_reduced1 = tsne1.fit_transform(embs.cpu())
    fig = plt.figure()

    ax = fig.add_subplot(121)
    for i in range(int(ranks.max())+1):
        idx = np.argwhere(ranks == i).flatten()
        y = np.zeros_like(embs_train_reduced1[idx]) + i
        ax.scatter(embs_train_reduced1[idx], y, color=colors2[i], label=str(i), s = 10)
        ax.scatter(embs_train_reduced1[idx].mean(), y.mean(), color=colors2[i], marker='*', s=150)



    ax2 = fig.add_subplot(122)
    # use rank as groups
    for i in range(int(ranks.max())+1):
        idx = np.argwhere(ranks == i).flatten()
        ax2.scatter(embs_train_reduced[idx, 0], embs_train_reduced[idx, 1], color=colors2[i], label=str(i), s = 10)
        ax2.scatter(embs_train_reduced[idx, 0].mean(), embs_train_reduced[idx, 1].mean(), color=colors2[i], marker='*', s=150)

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('rank')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax.set_title('1dim reduction')
    ax2.set_title('2dim reduction')
    ax.legend()
    ax2.legend()
    # tight layout
    plt.tight_layout()

    wandb.log({'embedding for ' + stage: wandb.Image(fig)}, step=epoch)



    return


def vis_embeddings(embs, ranks,
                        colors = ['red', 'orange',  'green', 'blue',  'purple'],
                   dim = 2, tsne = True):
    """transform reduce both embs to 2d and plot them together"""
    if tsne:
        if dim == 2:
            tsne = TSNE(init='pca', n_components=2, verbose=0, perplexity=40, n_iter=300)
        elif dim == 1:
            tsne = TSNE(init='pca', n_components=1, verbose=0, perplexity=40, n_iter=300)
        # ipdb.set_trace()
        embs_reduced = tsne.fit_transform(embs)
    else:
        # pca
        if dim == 2:
            pca = PCA(n_components=2)
        elif dim == 1:
            pca = PCA(n_components=1)
        embs_reduced = pca.fit_transform(embs)

    plot_emb(embs_reduced, ranks, colors, dim = dim)

    return


def vis_embeddings_imgs(ref_embs, query_embs, ref_ranks, query_ranks, ref_dataloader, query_dataloader,
                        colors = ['red', 'orange',  'green', 'blue',  'purple'],
                        img_size = 110,
                        num_img = 10,
                        ref_scale = 100,
                        query_scale = 100
                   ):
    """transform reduce both embs to 2d and plot them together

    plot some example images at corresponding locations

    """
    tsne = TSNE(init='pca', n_components=2, verbose=0, perplexity=40, n_iter=300)

    # concatenate embs
    embs = torch.cat([ref_embs, query_embs], dim=0)
    embs_reduced = tsne.fit_transform(embs.cpu())
    # deconcatenate
    ref_embs_reduced = embs_reduced[:len(ref_embs)]
    query_embs_reduced = embs_reduced[len(ref_embs):]

    # upscale both embs for plot
    ref_embs_reduced = ref_embs_reduced * ref_scale
    query_embs_reduced = query_embs_reduced * query_scale

    fig = plt.figure()
    ax = fig.add_subplot(121)


    # sample 5 images from each rank, plot image at corresponding location
    for i in range(int(ref_ranks.max())+1):
        idx = np.argwhere(ref_ranks == i).flatten()
        idx = np.random.choice(idx, num_img, replace=False)
        for j in range(len(idx)):
            img = ref_dataloader.dataset.imgs[idx[j]]
            img = plt.imread(img)[:,:,:3]
            ax.add_patch(Rectangle((ref_embs_reduced[idx[j], 0]-int((img_size+20)/2), ref_embs_reduced[idx[j], 1]-int((img_size+20)/2)), img_size+20, img_size+20, fill=True, color=colors[i], zorder = 2))
            ax.imshow(img, extent=(
            ref_embs_reduced[idx[j], 0] - int(img_size / 2), ref_embs_reduced[idx[j], 0] + int(img_size / 2),
            ref_embs_reduced[idx[j], 1] - int(img_size / 2), ref_embs_reduced[idx[j], 1] + int(img_size / 2)), zorder=3)

    # plot ref embs
    for i in range(int(ref_ranks.max()) + 1):
        idx = np.argwhere(ref_ranks == i).flatten()
        ax.scatter(ref_embs_reduced[idx, 0], ref_embs_reduced[idx, 1], color=colors[i], alpha = 0.5, label=str(i), s=10, zorder = 1)
        ax.scatter(ref_embs_reduced[idx, 0].mean(), ref_embs_reduced[idx, 1].mean(), color=colors[i], marker='*',
                   s=300, edgecolors='black', zorder = 4)



    ax12 = fig.add_subplot(122)
    # plot query embs
    for i in range(int(query_ranks.max())+1):
        idx = np.argwhere(query_ranks == i).flatten()
        idx = np.random.choice(idx, num_img, replace=False)
        for j in range(len(idx)):
            img = query_dataloader.dataset.imgs[idx[j]]
            print('rank: ', i, 'img: ', img)
            img = plt.imread(img)[:,:,:3]
            ax12.add_patch(Rectangle((query_embs_reduced[idx[j], 0]-int((img_size+20)/2), query_embs_reduced[idx[j], 1]-int((img_size+20)/2)), img_size+20, img_size+20, fill=True, color=colors[i], zorder = 2))
            ax12.imshow(img, extent=(
            query_embs_reduced[idx[j], 0] - int(img_size / 2), query_embs_reduced[idx[j], 0] + int(img_size / 2),
            query_embs_reduced[idx[j], 1] - int(img_size / 2), query_embs_reduced[idx[j], 1] + int(img_size / 2)), zorder=3)

    for i in range(int(query_ranks.max())+1):
        idx = np.argwhere(query_ranks == i).flatten()
        ax12.scatter(query_embs_reduced[idx, 0], query_embs_reduced[idx, 1], color=colors[i], alpha = 0.5, s=10, zorder = 1)
        ax12.scatter(query_embs_reduced[idx, 0].mean(), query_embs_reduced[idx, 1].mean(), color=colors[i], marker='*',
                   s=300, edgecolors='black', zorder = 4)


    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_title('train')
    ax12.set_xlabel('Dim 1')
    ax12.set_ylabel('Dim 2')
    ax12.set_title('test')

    ax.legend()

    plt.show()
    return



def vis_embeddings_imgs_vanilla(query_embs, query_ranks, query_labels, query_preds, query_dataloader, test_idx,
                        img_size = 110,
                        num_img = 10,
                        query_scale = 100,
                                font = 16,
                                offset = 40,
                                linspacing = 500,
                   ):
    from matplotlib.patches import Rectangle

    """transform reduce both embs to 2d and plot them together

    plot some example images at corresponding locations

    """
    tsne = TSNE(init='pca', n_components=2, verbose=0, perplexity=40, n_iter=500)
    embs_reduced = tsne.fit_transform(query_embs.cpu())

    # upscale both embs for plot
    query_embs_reduced = embs_reduced * query_scale

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    # plot query embs

    # sample based on query_embs_reduced position: so that every 10 meter grid there is a sample
    idx = []
    for i in range(int(query_embs_reduced[:, 0].min()), int(query_embs_reduced[:, 0].max()), linspacing):
        for j in range(int(query_embs_reduced[:, 1].min()), int(query_embs_reduced[:, 1].max()), linspacing):
            idx.append(np.argmin(np.linalg.norm(query_embs_reduced - np.array([i, j]), axis=1)))
    idx = np.array(idx)

    for j in range(len(idx)):
        img = query_dataloader.dataset.imgs[test_idx[idx[j]]]
        img = plt.imread(img)[:,:,:3]
        ax.add_patch(Rectangle((query_embs_reduced[idx[j], 0]-int((img_size+20)/2), query_embs_reduced[idx[j], 1]-int((img_size+20)/2)), img_size+10, img_size+10, fill=True, color='gray', zorder = 2))
        # move around if the position is already occupied
        extent = (query_embs_reduced[idx[j], 0] - int(img_size / 2), query_embs_reduced[idx[j], 0] + int(img_size / 2),
                    query_embs_reduced[idx[j], 1] - int(img_size / 2), query_embs_reduced[idx[j], 1] + int(img_size / 2))
        # # if the extent is already occupied, move around
        # while ax.transData.transform(extent[:2])[0] < 0 or ax.transData.transform(extent[2:])[1] < 0:
        #     extent = (extent[0] + 1, extent[1] + 1, extent[2] + 1, extent[3] + 1)
        ax.imshow(img, extent=extent, zorder=3)

        # ax.imshow(img, extent=(
        # query_embs_reduced[idx[j], 0] - int(img_size / 2), query_embs_reduced[idx[j], 0] + int(img_size / 2),
        # query_embs_reduced[idx[j], 1] - int(img_size / 2), query_embs_reduced[idx[j], 1] + int(img_size / 2)), zorder=3)
        # add text on bottom right
        ax.text(query_embs_reduced[idx[j], 0] + int(img_size / 2) + 5, query_embs_reduced[idx[j], 1] - int(img_size / 2), str(int(query_labels[idx[j]])), fontsize=font, color='royalblue', zorder=4, weight = 'bold')
        ax.text(query_embs_reduced[idx[j], 0] + int(img_size / 2) + 5, query_embs_reduced[idx[j], 1] - int(img_size / 2) + offset, str(int(query_preds[idx[j]])), fontsize=font, color='firebrick', zorder=4, weight = 'bold')


    # plot query embs
    labels_rd = (query_labels - np.percentile(query_labels, 1)) / (np.percentile(query_labels, 99) - np.percentile(query_labels, 1))
    # cut off
    labels_rd[labels_rd < 0] = 0
    labels_rd[labels_rd > 1] = 1
    ax.scatter(query_embs_reduced[:, 0], query_embs_reduced[:, 1], c=labels_rd, alpha = 0.9, s=5, zorder = 1, cmap='jet')

    ax.legend()
    # tick off
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    return



def log_init_pse_labels(preds_all, preds_rank_all, un_proxy_labels, epoch=0):
    # log two scatter plots
    # 1. proxy labels vs. preds_all
    # 2. proxy labels vs. preds_rank_all

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(un_proxy_labels, preds_all, s=10)
    ax.set_xlabel('proxy labels')
    ax.set_ylabel('preds')
    ax.set_title('unlabeled estimation')
    # regression line
    z = np.polyfit(un_proxy_labels, preds_all, 1)
    p = np.poly1d(z)
    r = np.corrcoef(un_proxy_labels, preds_all)[0, 1]
    ax.plot(un_proxy_labels, p(un_proxy_labels), "r--", label=f'y={z[0]:.2f}x+{z[1]:.2f}, r={r:.2f}')
    ax.legend()

    print('unlabeled estimation: regression between proxy label and value estimate: \ny={:.2f}x+{:.2f}, r={:.2f}'.format(z[0], z[1], r))

    ax2 = fig.add_subplot(122)
    ax2.scatter(un_proxy_labels, preds_rank_all, s=10)
    ax2.set_xlabel('proxy labels')
    ax2.set_ylabel('preds_rank')
    ax2.set_title('unlabeled rank estimation')
    # regression line
    z = np.polyfit(un_proxy_labels, preds_rank_all, 1)
    p = np.poly1d(z)
    r = np.corrcoef(un_proxy_labels, preds_rank_all)[0, 1]
    ax2.plot(un_proxy_labels, p(un_proxy_labels), "r--", label=f'y={z[0]:.2f}x+{z[1]:.2f}, r={r:.2f}')
    ax2.legend()
    print('unlabeled rank estimation: regression between proxy label and rank estimate: \ny={:.2f}x+{:.2f}, r={:.2f}'.format(z[0], z[1], r))
    wandb.log({'unlabeled estimation': wandb.Image(fig)}, step=epoch)

    return


def log_pse_ranks(pse_ranks, un_proxy_labels, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(un_proxy_labels, pse_ranks, s=10)
    ax.set_xlabel('proxy labels')
    ax.set_ylabel('pseudo ranks')
    ax.set_title('unlabeled rank estimation')
    # regression line
    z = np.polyfit(un_proxy_labels, pse_ranks, 1)
    p = np.poly1d(z)
    r = np.corrcoef(un_proxy_labels, pse_ranks)[0, 1]
    ax.plot(un_proxy_labels, p(un_proxy_labels), "r--", label=f'y={z[0]:.2f}x+{z[1]:.2f}, r={r:.2f}')
    ax.legend()
    wandb.log({'unlabeled rank estimation': wandb.Image(fig)}, step=epoch)

    return

def sample_fdcs(model, fdc_pts, train_labels, cfg):
    to_select = np.unique(train_labels)
    model.select_reference_points(to_select.astype(np.int32), fdc_pts)
    cfg.fiducial_point_num = len(to_select)
    return model, cfg


def print_network(config, model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        if p.requires_grad == True:
            num_params += p.numel()
    # print(model)
    # import ipdb
    # # ipdb.set_trace()
    if config.model == 'GOL':
        model_stats = summary(model,
                                  input_size=(config.batch_size*2, config.inputchannel, config.inputlength, config.inputlength),
                                  col_names=("input_size", "output_size", "num_params"))
    elif config.model == 'regressor':
        model_stats = summary(model,
                              input_size=(
                              config.batch_size, config.inputchannel, config.inputlength, config.inputlength),
                              col_names=("input_size", "output_size", "num_params"))
    #except:
    #model_stats = summary(model)




def drr(embeds, centroids, ranks, beta):
    """
    embeds: (n, d)
    centroids: (k, d)
    ranks: (k,) - rank of each centroid
    beta: float
    """
    # sort centroids by rank
    centroids = centroids[ranks.argsort()]
    # assign each embedding to the closest centroid
    cd = np.linalg.norm(embeds[:, None] - centroids[None, :], axis=-1)
    # find closest centroid
    closest_centroid = np.argmin(cd, axis=-1)

    # iterate over centroids
    num = 0
    for i in range(1, centroids.shape[0]):
        weight = np.power(i - np.arange(i), beta)
        # average weighted distance from this centroid to centroids with lower rank
        sum_d = np.sum(np.linalg.norm(centroids[i][None, :] - centroids[:i][None, :], axis=-1) * weight, axis=-1)
        num += sum_d.mean() / weight.sum()

    den = 0
    for i in range(centroids.shape[0]):
        # collect average distance of embeddings assigned to this centroid
        assigned = embeds[closest_centroid == i]
        this_centroid_cdists = np.linalg.norm(assigned[:, None] - assigned[None, :], axis=-1)
        den += np.mean(this_centroid_cdists[np.triu_indices(assigned.shape[0], k=1)])

    return num / den