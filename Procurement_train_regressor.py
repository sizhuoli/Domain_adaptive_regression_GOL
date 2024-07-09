import os
import time
import sys
from copy import deepcopy

import numpy as np
from sklearn.metrics import r2_score

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('Agg')
from config.procurement_base import ConfigBasic
from utils.util import write_log, log_configs, save_ckpt, set_wandb, print_network, \
    log_eval_preds_vanilla
from utils.util import AverageMeter
from networks.util import prepare_model

def main(cfg):
    #****************** direct regressor training ******************
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')
    from data.get_dataset_base_fixed import get_datasets

    # dataloader
    loader_dict = get_datasets(cfg)
    cfg.n_ranks = cfg.class_number

    # model
    model = prepare_model(cfg)

    if cfg.resume_model:
        model.load_state_dict(torch.load(cfg.resume_model_path)['model'])
        print(f'[*] ######################### model loaded from {cfg.resume_model_path}')


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

    if cfg.loss_func == 'MAE':
        criterion = nn.L1Loss()
    elif cfg.loss_func == 'MSE':
        criterion = nn.MSELoss()

    else:
        raise NotImplementedError


    if torch.cuda.is_available():
        if cfg.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    val_r2_best = -1000
    log_dict = dict()
    loss_record = dict()
    for epoch in range(cfg.epochs):
        print("==> training...")

        time1 = time.time()
        train_loss, loss_record = train(epoch, loader_dict['train'], model, optimizer, criterion, cfg,
                                        prev_loss_record=loss_record)

        if cfg.scheduler:
            scheduler.step()
        time2 = time.time()
        print('epoch {}, loss {:.4f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))

        if epoch % cfg.val_freq == 0:
            print('==> validation...')
            val_mae, val_r2 = validate(loader_dict, model, epoch, val_r2_best)

            if val_r2 > val_r2_best:
                val_r2_best = val_r2
                save_ckpt(cfg, model, optimizer, epoch, train_loss,
                          None, None, None, None, f'val_best_r2.pth')

        if epoch % cfg.test_freq == 0:
            print('==> testing...')
            test_mae, test_r2 = test(loader_dict, model)

        if cfg.wandb:
            log_dict['Train Loss'] = train_loss
            log_dict['Val Mae'] = val_mae
            log_dict['Val R2'] = val_r2
            log_dict['LR'] = scheduler.get_last_lr()[0] if scheduler else cfg.learning_rate
            log_dict['Test Mae'] = test_mae
            log_dict['Test R2'] = test_r2
            wandb.log(log_dict)

    print('[*] Training ends')

def train(epoch, train_loader, model, optimizer, criterion , cfg, prev_loss_record):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    r2s = AverageMeter()

    loss_record = deepcopy(prev_loss_record)
    end = time.time()

    for idx, (img, targets, _) in enumerate(train_loader):
        print(f'Epoch {epoch}, batch {idx}', end='\r')

        if torch.cuda.is_available():
            img = img.cuda()
            targets = targets.cuda()

        data_time.update(time.time() - end)
        # ===================forward=====================
        output = model.encoder(img).squeeze()
        # ==================== loss======================
        loss = criterion(output, targets)
        losses.update(loss.item(), output.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================meters=====================
        r2 = r2_score(targets.cpu().detach().numpy(), output.cpu().detach().numpy())
        r2s.update(r2, output.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % cfg.print_freq == 0:
            write_log(cfg.logfile,
                      f'Epoch [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f}\t'
                      f'Data {data_time.val:3f}\t'
                      f'Loss {losses.val:.4f}\t'
                        f'R2 {r2s.val:.4f}\t')
            sys.stdout.flush()
    return losses.avg, loss_record


def validate(loader_dict, model, epoch, val_r2_best):
    model.eval()
    preds_all = []
    targets_all = []
    for idx, (x_base, cur_target, _, _) in enumerate(loader_dict['val']):
        if torch.cuda.is_available():
            x_base = x_base.cuda()
        with torch.no_grad():
            output = model.encoder(x_base).squeeze()
        preds_all.append(output.cpu().detach().numpy())
        targets_all.append(cur_target)
    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    cur_r2 = r2_score(targets_all, preds_all)
    cur_mae = np.mean(np.abs(targets_all - preds_all))

    if cur_r2 > val_r2_best:
        log_eval_preds_vanilla(preds_all, targets_all, epoch)

    sys.stdout.flush()
    return cur_mae, cur_r2

def test(loader_dict, model):
    model.eval()
    preds_all = []
    targets_all = []
    for idx, (x_base, cur_target, _, _) in enumerate(loader_dict['test']):
        if torch.cuda.is_available():
            x_base = x_base.cuda()
        with torch.no_grad():
            output = model.encoder(x_base).squeeze()
        preds_all.append(output.cpu().detach().numpy())
        targets_all.append(cur_target)
    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    cur_r2 = r2_score(targets_all, preds_all)
    cur_mae = np.mean(np.abs(targets_all - preds_all))

    sys.stdout.flush()
    return cur_mae, cur_r2


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
    parser = argparse.ArgumentParser(description='Train a direct regressor')
    parser.add_argument('--target', type=str, default='height', help='target attribute')
    args = parser.parse_args()

    cfg = ConfigBasic(args.target)
    main(cfg)
