# few shot fine tune embedding features for cross domain
# adapt from train-GOL
import os
import sys
import time
from copy import deepcopy

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb

from utils.sam import SAM

matplotlib.use('Agg')
from config.cross_domain_finetune_GOL import Config
from utils.util import write_log, log_configs, set_wandb, print_network, query_diffusion_head

from utils.util import AverageMeter, ClassWiseAverageMeter, cls_accuracy, extract_embs
from utils.loss_util import compute_order_loss, compute_metric_loss, compute_center_loss
from networks.util import prepare_model



def main(cfg):
    if cfg.train_loss_scheme == 'gol':
        finetune_gol_loss(cfg)
    else:
        raise NotImplementedError





def finetune_gol_loss(cfg):
    # gird search to find best lr and then finetune
    print('starting grid search')
    global_best_val_r2 = -99
    for cur_lr in cfg.lr_list:
        best_val_r2, best_num_itr = hyper_select(cfg, cur_lr)
        if best_val_r2 > global_best_val_r2:
            global_best_val_r2 = best_val_r2
            best_lr = cur_lr
            best_itr = best_num_itr
    print(f'[*] Best lr: {best_lr}, Best Val R2: {global_best_val_r2:.4f}, Best Num Itr: {best_itr}')
    # repeart 3 times
    for rep in range(cfg.repeat):
        # finetune using best lr
        model, optimizer, loader_dict, scheduler = prepare_config(cfg, mode='finetune', lr=best_lr)
        train_loader = loader_dict['train']
        train_for_test_loader = loader_dict['train_for_val']
        test_loader = loader_dict['test']
        log_dict = dict()
        # log best itr
        if cfg.wandb:
            wandb.log({'Best Number Itr': best_itr})
        if best_itr == 0:
            best_itr = cfg.iterations
        print(f'[*] Total num Itr: {best_itr}')
        # test without adapt
        test_r2_init, test_f1_init = test(train_for_test_loader, test_loader, model, cfg)
        if cfg.wandb:
            wandb.log({'Test R2 Init': test_r2_init, 'Test F1 Init': test_f1_init})

        loss_record = dict()
        loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]
        for itr in range(best_itr):
            train_iterator = iter(train_loader)
            train_for_test_iterator = iter(train_for_test_loader)
            data = next(train_iterator)

            loss, loss_record, angle_loss, dist_loss, center_loss = train_one_itr(data, model, optimizer, cfg, itr, prev_loss_record=loss_record)
            if itr % cfg.test_freq == 0:
                test_r2, test_f1 = test(train_for_test_iterator, test_loader, model, cfg)
                if itr % cfg.print_freq == 0:
                    print(f'[*] Test R2: {test_r2:.4f}, Test F1: {test_f1:.4f}')

            if cfg.wandb:
                log_dict['Train Loss'] = loss
                log_dict['Angle Loss'] = angle_loss
                log_dict['Dist Loss'] = dist_loss
                log_dict['Center Loss'] = center_loss
                log_dict['Test R2'] = test_r2
                log_dict['Test F1'] = test_f1
                wandb.log(log_dict)

        if cfg.wandb:
            wandb.finish()
        sys.stdout.flush()


def hyper_select(cfg, lr):
    #******************  grid search ******************
    model, optimizer, loader_dict, scheduler = prepare_config(cfg, mode='hyperselect', lr=lr)

    # following finetuning paper
    # train
    train_loader = loader_dict['train']
    train_for_test_loader = loader_dict['train_for_val']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']

    best_val_r2 = -99
    best_num_itr = None
    log_dict = dict()
    # test without adapt
    test_r2, test_f1 = test(train_for_test_loader, test_loader, model, cfg)
    print('before finetuning====================')
    print(f'[*] Test R2: {test_r2:.4f}, Test F1: {test_f1:.4f}')


    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]
    for itr in range(cfg.iterations):
        train_iterator = iter(train_loader)
        train_for_test_iterator = iter(train_for_test_loader)
        val_iterator = iter(val_loader)
        data = next(train_iterator)
        # ipdb.set_trace()
        train_loss, loss_record, angle_loss, dist_loss, center_loss = train_one_itr(data, model, optimizer, cfg, itr, prev_loss_record=loss_record)
        if itr % cfg.val_freq == 0:
            val_r2, val_f1 = test(train_for_test_iterator, val_iterator, model, cfg)
            if itr % cfg.print_freq == 0:
                print(f'[*] itr: {itr}, Val R2: {val_r2:.4f}, Val F1: {val_f1:.4f}')
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_num_itr = itr


        if cfg.wandb:
            log_dict['Train Loss'] = train_loss
            log_dict['Angle Loss'] = angle_loss
            log_dict['Dist Loss'] = dist_loss
            log_dict['Center Loss'] = center_loss
            log_dict['Val R2'] = val_r2
            log_dict['Val F1'] = val_f1
            wandb.log(log_dict)

    if cfg.wandb:
        wandb.finish()
    # flush
    sys.stdout.flush()
    return best_val_r2, best_num_itr


def prepare_config(cfg, mode, lr):
    torch.cuda.empty_cache()
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    from data.get_dataset_adapt_finetune_GOL import get_datasets
    cfg.n_ranks = cfg.infer_class_number

    loader_dict = get_datasets(cfg, mode=mode) # subset fewshot data into train and val

    model = prepare_model(cfg)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(cfg.model_path)['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'[*] ######################### weights loaded from {cfg.model_path}')


    if cfg.freeze_bn:
        freeze_bn(model)
        print('[*] ****************** BatchNorm frozen.')



    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True


    param = model.parameters() # end2end finetune

    if cfg.SAM:
        if cfg.optim == 'adam':
            optimizer = SAM(param, torch.optim.Adam, lr=lr,
                            weight_decay=cfg.weight_decay, rho=cfg.rho)
        else:
            optimizer = SAM(param, torch.optim.SGD, lr=lr,
                            weight_decay=cfg.weight_decay, momentum=cfg.momentum, nesterov=True,
                            rho=cfg.rho)
    else:
        if cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param, lr=lr,
                                         weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.SGD(param, lr=lr,
                                        weight_decay=cfg.weight_decay, momentum=cfg.momentum, nesterov=True)


    if cfg.wandb:
        cfg.learning_rate = lr
        if 'lr' in cfg.experiment_name:
            # update the lr in experiment name
            prev_lr = cfg.experiment_name.split('_')[-1]
            cfg.experiment_name = cfg.experiment_name.replace(prev_lr, str(lr))
        else:
            cfg.experiment_name = f'{cfg.experiment_name}_lr_{lr}'
        if mode == 'finetune':
            if 'final_finetune' not in cfg.experiment_name:
                cfg.experiment_name = f'{cfg.experiment_name}_final_finetune'
        set_wandb(cfg)
        wandb.watch(model)

    scheduler = None

    print_network(cfg, model)
    return model, optimizer, loader_dict, scheduler






def get_pairs_equally(ranks, tau, m=32):
    orders = []
    base_idx = []
    ref_idx = []
    N = len(ranks)
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand(1) > 0.5:
                base_idx.append(i)
                ref_idx.append(j)
                order_ij = get_order_labels(ranks[i], ranks[j], tau)
                orders.append(order_ij)
            else:
                base_idx.append(j)
                ref_idx.append(i)
                order_ji = get_order_labels(ranks[j], ranks[i], tau)
                orders.append(order_ji)
    refine = []
    orders = np.array(orders)
    for o in range(3):
        o_idxs = np.argwhere(orders == o).flatten()
        if len(o_idxs) > m:
            sel = np.random.choice(o_idxs, m, replace=False)
            refine.append(sel)
        else:
            refine.append(o_idxs)
    refine = np.concatenate(refine)
    base_idx = np.array(base_idx)[refine]
    ref_idx = np.array(ref_idx)[refine]
    orders = orders[refine]
    # ipdb.set_trace()
    return base_idx, ref_idx, orders


def get_order_labels(rank_base, rank_ref, tau):
    if rank_base > rank_ref + tau:
        order = 0
    elif rank_base < rank_ref - tau:
        order = 1
    else:
        order = 2
    return order




def train_one_itr(data, model, optimizer, cfg, itr, prev_loss_record):
    """One itr training"""
    model.train()

    losses = AverageMeter()
    angle_losses = AverageMeter()
    dist_losses = AverageMeter()
    center_losses = AverageMeter()
    angle_acc_meter = ClassWiseAverageMeter(2)
    loss_record = deepcopy(prev_loss_record)


    end = time.time()
    x_base, x_ref, _, ranks, _ = data

    ranks_np = torch.cat(ranks).detach().numpy()  # [base_rank, ref_rank]
    base_idx, ref_idx, order_labels = get_pairs_equally(ranks_np, cfg.tau)

    if torch.cuda.is_available():
        x_base = x_base.cuda()
        x_ref = x_ref.cuda()

    # ===================forward=====================
    embs = model.encoder(torch.cat([x_base, x_ref], dim=0))

    # =====================gol loss======================
    dist_loss = compute_metric_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg.margin, cfg)
    angle_loss, logits, order_gt = compute_order_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg)
    center_loss = compute_center_loss(embs, ranks_np, model.ref_points, cfg)
    total_loss = (cfg.drct_wieght * angle_loss) + (cfg.metric_los_wei * dist_loss) + (cfg.center_los_wei * center_loss)

    # =====================prototype loss======================

    losses.update(total_loss.item(), x_base.size(0))
    angle_losses.update(angle_loss.item(), x_base.size(0))
    dist_losses.update(dist_loss.item(), x_base.size(0))
    center_losses.update(center_loss.item(), x_base.size(0))
    # ===================backward=====================

    if cfg.SAM:
        total_loss.backward()
        optimizer.first_step(zero_grad=True)
        embs = model.encoder(torch.cat([x_base, x_ref], dim=0))
        dist_loss = compute_metric_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg.margin, cfg)
        angle_loss, logits, order_gt = compute_order_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg)
        center_loss = compute_center_loss(embs, ranks_np, model.ref_points, cfg)
        total_loss = (cfg.drct_wieght * angle_loss) + (cfg.metric_los_wei * dist_loss) + (
                    cfg.center_los_wei * center_loss)
        total_loss.backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    acc, cnt = cls_accuracy(nn.functional.softmax(logits, dim=-1), order_gt, n_cls=2)

    angle_acc_meter.update(acc, cnt)

    # ===================meters=====================
    if itr % cfg.print_freq == 0:
        write_log(cfg.logfile, f'[*] Train Loss: {losses.avg:.4f}, Train Angle Loss: {angle_losses.avg:.4f}, Train Dist Loss: {dist_losses.avg:.4f}, Train Center Loss: {center_losses.avg:.4f}')

    return losses.avg, loss_record, angle_losses.avg, dist_losses.avg, center_losses.avg




def test(train_for_test_loader, test_loader, model, cfg, return_mae = False, vanilla_loader = False):
    model.eval()
    embs_train, train_labels, train_ranks = extract_embs(model.encoder, train_for_test_loader, return_labels=True, vanilla_loader=vanilla_loader)
    embs_train = embs_train.cuda()

    embs_test, test_labels, test_ranks = extract_embs(model.encoder, test_loader, return_labels=True, vanilla_loader=vanilla_loader)
    embs_test = embs_test.cuda()

    mae, _, r2_diffusion, f1_diffusion = query_diffusion_head(embs_test, embs_train, train_labels, train_ranks,
                                        test_labels, test_ranks, cfg, None)

    sys.stdout.flush()
    if return_mae:
        return r2_diffusion, f1_diffusion, mae
    else:
        return r2_diffusion, f1_diffusion




def compute_centroids(embs, ranks, n_cls):
    centroids = torch.zeros([n_cls, embs.shape[1]]).cuda()
    for i_cls in range(n_cls):
        # ipdb.set_trace()
        centroids[i_cls] = torch.mean(embs[ranks == i_cls], dim=0)
    return centroids

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # set parser with input from command line
    import argparse
    parser = argparse.ArgumentParser(description='FInetune GOL')
    parser.add_argument('--source_dataset', type=str, default='DK', help='dataset name: DK')
    parser.add_argument('--infer_dataset', type=str, default='SP', help='dataset name: SP, slovenia, slovakia')
    parser.add_argument('--target', type=str, default='height', help='target name: height, count, treecover5m')
    parser.add_argument('--backbone', type=str, default='vit_b16_reduce', help='backbone name: vit_b16_reduce, vgg16v2norm_reduce')

    args = parser.parse_args()

    cfg = Config(args.source_dataset, args.infer_dataset, args.target, args.backbone)
    main(cfg)
