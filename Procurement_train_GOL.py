import os
import time
import sys
from copy import deepcopy

import numpy as np
from sklearn.metrics import r2_score, f1_score

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('Agg')
from config.procurement_base import ConfigBasic
from utils.util import write_log, log_configs, save_ckpt, set_wandb, print_network, \
    log_embeddings, process_embs_fixed_k_search, \
    log_eval_ranks_vanilla, log_eval_preds_vanilla
from utils.util import AverageMeter, ClassWiseAverageMeter, cls_accuracy, extract_embs
from utils.loss_util import compute_order_loss, compute_metric_loss, compute_center_loss
from utils.diffusion import alpha_query_expansion
from networks.util import prepare_model


def on_load_checkpoint(model, checkpoint: dict) -> None:
    state_dict = checkpoint["model"]
    model_state_dict = model.state_dict()
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
            else:
                model_state_dict[k] = state_dict[k]
                print(f"Loading parameter: {k}")
                is_changed = True
        else:
            print(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)


def main(cfg):
    #****************** GOL training ******************
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')
    from data.get_dataset_base_fixed import get_datasets
    # dataloader
    loader_dict = get_datasets(cfg)

    cfg.n_ranks = cfg.class_number

    # model
    model = prepare_model(cfg)

    if cfg.resume_model:
        on_load_checkpoint(model, torch.load(cfg.resume_model_path))


    if cfg.freeze_bn:
        freeze_bn(model)
        print('[*] ****************** BatchNorm frozen.')
    print_network(cfg, model)
    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(model)

    if cfg.adam:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg.learning_rate,
                              momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    if cfg.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.learning_rate * 0.001)
    elif cfg.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate)

    else:
        scheduler = None

    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    val_f1_best = -99
    val_f1_expanded_best = -99
    log_dict = dict()
    # init loss matrix
    loss_record = dict()
    loss_record['angle'] = [np.zeros([cfg.n_ranks, cfg.n_ranks]), np.zeros([cfg.n_ranks, cfg.n_ranks])]
    for epoch in range(cfg.epochs):
        print("==> training...")

        time1 = time.time()
        train_loss, loss_record, angle_loss, dist_loss, center_loss = train(epoch, loader_dict['train'], model, optimizer, cfg,
                                        prev_loss_record=loss_record)


        if cfg.scheduler:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        if epoch % cfg.val_freq == 0:
            print('==> validation...')

            val_f1, val_r2, val_mae, val_rank_acc, val_rank_acc_cls, \
                val_f1_expanded, val_r2_expended, embs_train, train_labels, train_ranks \
                = validate(loader_dict, model, cfg, epoch, val_f1_best)


            if val_f1 > val_f1_best:
                val_f1_best = val_f1
                save_ckpt(cfg, model, optimizer, epoch, train_loss, angle_loss,
                          dist_loss, center_loss, val_f1, f'val_best_rank_f1.pth')

            if val_f1_expanded > val_f1_expanded_best:
                val_f1_expanded_best = val_f1_expanded
                save_ckpt(cfg, model, optimizer, epoch, train_loss, angle_loss,
                          dist_loss, center_loss, val_f1_expanded, f'val_best_rank_f1_alpha_expanded.pth')

        if epoch % cfg.test_freq == 0:
            print('==> testing... for monitoring, no interaction with model training')
            try:
                test_f1, test_r2, test_f1_exp, test_r2_exp = test(loader_dict, model, cfg, epoch, embs_train, train_labels, train_ranks)

            except:
                print('test skipped')
                continue


        if cfg.wandb:
            log_dict['Train Loss'] = train_loss
            log_dict['Val R2'] = val_r2
            log_dict['Val rank f1'] = val_f1
            log_dict['LR'] = scheduler.get_last_lr()[0] if scheduler else cfg.learning_rate
            log_dict['Angle loss'] = angle_loss
            log_dict['Dist loss'] = dist_loss
            log_dict['Center loss'] = center_loss
            for i in range(cfg.n_ranks):
                log_dict[f'Val rank accuracy class {i}'] = val_rank_acc_cls[i]

            log_dict['Val rank f1 alpha expanded'] = val_f1_expanded
            log_dict['Val R2 alpha expanded'] = val_r2_expended
            try:
                log_dict['Test rank f1'] = test_f1
                log_dict['Test R2'] = test_r2
                log_dict['Test rank f1 alpha expanded'] = test_f1_exp
                log_dict['Test R2 alpha expanded'] = test_r2_exp
            except:
                pass

            wandb.log(log_dict)

    print('[*] Training ends')


def update_loss_matrix(A, loss, base_ranks, ref_ranks=None):
    batch_size = len(base_ranks)
    if ref_ranks is not None:
        for i in range(batch_size):
            A[0][base_ranks[i], ref_ranks[i]] += loss[i]
            A[1][base_ranks[i], ref_ranks[i]] += 1
    else:
        for i in range(batch_size):
            A[0][base_ranks[i]] += loss[i]
            A[1][base_ranks[i]] += 1
    return A


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


def train(epoch, train_loader, model, optimizer, cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_losses = AverageMeter()
    dist_losses = AverageMeter()
    center_losses = AverageMeter()
    angle_acc_meter = ClassWiseAverageMeter(2)

    loss_record = deepcopy(prev_loss_record)
    end = time.time()

    for idx, (x_base, x_ref, _, ranks, _, _) in enumerate(train_loader):
        print(f'Epoch {epoch}, batch {idx}', end='\r')
        ranks_np = torch.cat(ranks).detach().numpy()  # [base_rank, ref_rank]
        base_idx, ref_idx, order_labels = get_pairs_equally(ranks_np, cfg.tau)

        if torch.cuda.is_available():
            x_base = x_base.cuda()
            x_ref = x_ref.cuda()

        data_time.update(time.time() - end)

        # ===================forward=====================
        embs = model.encoder(torch.cat([x_base, x_ref], dim=0))

        # =====================gol loss======================
        tic = time.time()
        dist_loss = compute_metric_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg.margin, cfg)
        tic = time.time()
        angle_loss, logits, order_gt = compute_order_loss(embs, base_idx, ref_idx, ranks_np, model.ref_points, cfg)
        center_loss = compute_center_loss(embs, ranks_np, model.ref_points, cfg)
        total_loss = (cfg.drct_wieght * angle_loss) + (cfg.metric_los_wei * dist_loss) + (cfg.center_los_wei * center_loss)
        losses.update(total_loss.item(), x_base.size(0))
        angle_losses.update(angle_loss.item(), x_base.size(0))
        dist_losses.update(dist_loss.item(), x_base.size(0))
        center_losses.update(center_loss.item(), x_base.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc, cnt = cls_accuracy(nn.functional.softmax(logits, dim=-1), order_gt, n_cls=2)
        angle_acc_meter.update(acc, cnt)
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'
                      f'Angle-Loss {angle_losses.val:.4f}\t'
                      f'Dist-Loss {dist_losses.val:.4f}\t'
                      f'Center-Loss {center_losses.val:.4f}\t'
                      f'Angle-Acc [{angle_acc_meter.val[0]:.3f}  {angle_acc_meter.val[1]:.3f}]  [{angle_acc_meter.total_avg:.3f}]\t'
                      )
            sys.stdout.flush()
    write_log(cfg.logfile,
              f' * Angle-Acc [{angle_acc_meter.avg[0]:.3f}  {angle_acc_meter.avg[1]:.3f}]  [{angle_acc_meter.total_avg:.3f}]\n')

    return losses.avg, loss_record, angle_losses.avg, dist_losses.avg, center_losses.avg


def validate(loader_dict, model, cfg, epoch, val_score_best):
    model.eval()
    data_time = AverageMeter()
    embs_train, train_labels, train_ranks = extract_embs(model.encoder, loader_dict['train_for_val'], return_labels=True)
    embs_train = embs_train.cuda()

    embs_test, test_labels, test_ranks = extract_embs(model.encoder, loader_dict['val'], return_labels=True)
    embs_test = embs_test.cuda()
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))

    preds_all, preds_rank_all = process_embs_fixed_k_search(embs_test, embs_train, train_labels, train_ranks,
                                                            data_time, n_test, n_batch, cfg,
                                                            rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                            rec_l=cfg.reciprocal_l)


    _, f1_score_mic = log_eval_ranks_vanilla(preds_rank_all, test_ranks, epoch)
    r2 = r2_score(test_labels, preds_all)
    mae = np.mean(np.abs(test_labels - preds_all))

    if f1_score_mic > val_score_best:
        log_embeddings(embs_train, train_labels, epoch, train_ranks, stage='train-labels')
        log_embeddings(embs_test, test_labels, epoch, test_ranks, stage='valid-labels')
        log_eval_preds_vanilla(preds_all, test_labels, epoch)
    rank_acc = np.mean(np.array(preds_rank_all) == np.array(test_ranks))
    rank_acc_cls = []
    for i in range(cfg.n_ranks):
        rank_acc_cls.append(np.mean(np.array(preds_rank_all)[np.array(test_ranks) == i] == i))
    rank_acc_cls = np.array(rank_acc_cls)


    embs_test_exp = alpha_query_expansion(embs_test, alpha=cfg.alpha, n=cfg.alpha_n)
    embs_train_exp = alpha_query_expansion(embs_train, alpha=cfg.alpha, n=cfg.alpha_n)
    preds_all_exp, preds_rank_all_exp = process_embs_fixed_k_search(embs_test_exp, embs_train_exp, train_labels, train_ranks,
                                                            data_time, n_test, n_batch, cfg,
                                                            rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                            rec_l=cfg.reciprocal_l)
    f1_score_mic_exp = f1_score(test_ranks, preds_rank_all_exp, average='micro')
    r2_exp = r2_score(test_labels, preds_all_exp)


    sys.stdout.flush()
    return f1_score_mic, r2, mae, rank_acc, rank_acc_cls, f1_score_mic_exp, r2_exp, embs_train, train_labels, train_ranks


def test(loader_dict, model, cfg, epoch, embs_train, labels_train, ranks_train):
    model.eval()
    data_time = AverageMeter()
    embs_test, test_labels, test_ranks = extract_embs(model.encoder, loader_dict['test'], return_labels=True)
    embs_test = embs_test.cuda()
    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.batch_size))

    preds_all, preds_rank_all = process_embs_fixed_k_search(embs_test, embs_train, labels_train, ranks_train,
                                                            data_time, n_test, n_batch, cfg,
                                                            rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                            rec_l=cfg.reciprocal_l)

    log_eval_preds_vanilla(preds_all, test_labels, epoch, set = 'testset')
    _, f1 = log_eval_ranks_vanilla(preds_rank_all, test_ranks, epoch, set = 'testset')
    r2 = r2_score(test_labels, preds_all)

    # alpha expansion
    embs_test_exp = alpha_query_expansion(embs_test, alpha=cfg.alpha, n=cfg.alpha_n)
    embs_train_exp = alpha_query_expansion(embs_train, alpha=cfg.alpha, n=cfg.alpha_n)
    preds_all_exp, preds_rank_all_exp = process_embs_fixed_k_search(embs_test_exp, embs_train_exp, labels_train, ranks_train,
                                                            data_time, n_test, n_batch, cfg,
                                                            rec_k1=cfg.reciprocal_k1, rec_k2=cfg.reciprocal_k2,
                                                            rec_l=cfg.reciprocal_l)
    f1_exp = f1_score(test_ranks, preds_rank_all_exp, average='micro')
    r2_exp = r2_score(test_labels, preds_all_exp)


    sys.stdout.flush()
    return f1, r2, f1_exp, r2_exp



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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='height')
    args = parser.parse_args()

    cfg = ConfigBasic(args.target)
    main(cfg)
