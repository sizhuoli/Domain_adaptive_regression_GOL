# few shot fine tune regressor for cross domain
# adapt from Deployment_finetune_GOL.py
import os
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
from sklearn.metrics import mean_absolute_error, r2_score

from utils.sam import SAM

matplotlib.use('Agg')
from config.cross_domain_finetune_regressor import Config
from utils.util import write_log, to_np, log_configs, set_wandb, print_network
from utils.util import AverageMeter
from networks.util import prepare_model


def main(cfg):
    # gird search to find best lr and then finetune
    print('starting grid search')
    global_best_val_loss = 1e10
    for cur_lr in cfg.lr_list:
        best_val_loss, best_num_itr = hyper_select(cfg, cur_lr)
        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            best_lr = cur_lr
            best_itr = best_num_itr
    print(f'[*] Best lr: {best_lr}, Best Val Loss: {global_best_val_loss:.4f}, Best Num Itr: {best_itr}')
    # repeart 3 times
    for rep in range(cfg.repeat):
        torch.cuda.empty_cache()
        # finetune using best lr
        model, optimizer, criterion, loader_dict, scheduler = prepare_config(cfg, mode='finetune', lr=best_lr)
        train_loader = loader_dict['train']
        test_loader = loader_dict['test']
        log_dict = dict()
        # log best itr
        if cfg.wandb:
            wandb.log({'Best Number Itr': best_itr})
        if best_itr == 0:
            best_itr = cfg.iterations
        print(f'[*] Total num Itr: {best_itr}')
        # test without adapt
        test_r2_init, test_mae_init, test_rmse_init = test(test_loader, model)
        if cfg.wandb:
            wandb.log({'Test R2 Init': test_r2_init, 'Test rmse Init': test_rmse_init})
        for itr in range(best_itr):
            try:
                train_iterator = iter(train_loader)
                x, y, _ = next(train_iterator)

                loss, r2 = train_one_itr(x, y, model, optimizer, criterion, cfg, itr)
                if itr % cfg.test_freq == 0:
                    test_r2, test_mae, test_rmse = test(test_loader, model)
                    print(f'[*] Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
                    if cfg.wandb:
                        log_dict['Train Loss'] = loss
                        log_dict['Train R2'] = r2
                        log_dict['Test R2'] = test_r2
                        log_dict['Test MAE'] = test_mae
                        log_dict['Test RMSE'] = test_rmse
                        wandb.log(log_dict)
            except:
                print('encounter error, ignore this itr')
                continue
            # release process memory
            torch.cuda.empty_cache()

        if cfg.wandb:
            wandb.finish()
        # flush
        sys.stdout.flush()



def hyper_select(cfg, lr):
    #******************  grid search ******************
    model, optimizer, criterion, loader_dict, scheduler = prepare_config(cfg, mode='hyperselect', lr=lr)

    # following finetuning paper
    # train
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']

    best_val_loss = 1e10
    best_num_itr = None
    log_dict = dict()
    # test without adapt
    test_r2, test_mae, test_rmse = test(test_loader, model)
    print('before finetuning====================')
    print(f'[*] Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
    for itr in range(cfg.iterations):
        try:
            train_iterator = iter(train_loader)
            val_iterator = iter(val_loader)
            x, y, _ = next(train_iterator)
            # ipdb.set_trace()
            loss, r2 = train_one_itr(x, y, model, optimizer, criterion, cfg, itr)
            if itr % cfg.val_freq == 0:
                val_r2, val_mae, val_rmse = test(val_iterator, model)
                if itr % cfg.print_freq == 0:
                    print(f'[*] itr: {itr}, Val R2: {val_r2:.4f}, Val MAE: {val_mae:.4f}')
                if val_mae < best_val_loss:
                    best_val_loss = val_mae
                    best_num_itr = itr


            if cfg.wandb:
                log_dict['Train Loss'] = loss
                log_dict['Train R2'] = r2
                log_dict['Val R2'] = val_r2
                log_dict['Val MAE'] = val_mae
                log_dict['Val RMSE'] = val_rmse
                wandb.log(log_dict)
        except:
            print('encounter error, ignore this itr')
            continue

    # finish wandb
    if cfg.wandb:
        wandb.finish()
    return best_val_loss, best_num_itr

def prepare_config(cfg, mode, lr):
    torch.cuda.empty_cache()
    cfg.logfile = log_configs(cfg, log_file='train_log.txt')

    from data.get_dataset_adapt_finetune_regressor import get_datasets

    # dataloader
    loader_dict = get_datasets(cfg, mode=mode) # subset fewshot data into train and val

    # model
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
    if cfg.loss_fn == 'mae':
        criterion = nn.L1Loss()
    elif cfg.loss_fn == 'mse':
        criterion = nn.MSELoss()

    print_network(cfg, model)
    return model, optimizer, criterion, loader_dict, scheduler



def test(data_loader, model):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for idx, (x, y, _) in enumerate(data_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            pred = model.encoder(x)
            preds.append(pred)
            labels.append(y)

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    r2 = r2_score(to_np(labels), to_np(preds))
    mae = mean_absolute_error(to_np(labels), to_np(preds))
    # rmse
    rmse = np.sqrt(np.mean((to_np(labels) - to_np(preds))**2))
    # flash
    sys.stdout.flush()
    return r2, mae, rmse



def train_one_itr(x, y, model, optimizer, criterion, cfg, itr):
    """One itr training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    r2_meter = AverageMeter()
    end = time.time()
    x = x.cuda()
    y = y.cuda()
    pred = model.encoder(x)
    loss = criterion(pred, y)
    if cfg.SAM:
        loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model.encoder(x), y).backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    batch_time.update(time.time() - end)
    r2 = r2_score(to_np(y), to_np(pred))
    r2_meter.update(r2, x.size(0))
    losses.update(loss.item(), x.size(0))
    if itr % cfg.print_freq == 0:
        write_log(cfg.logfile, f'[*] Train Loss: {losses.avg:.4f}, Train R2: {r2_meter.avg:.4f}')
    # flush
    sys.stdout.flush()
    return losses.avg, r2_meter.avg




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
    parser = argparse.ArgumentParser(description='Finetune Regressor')
    parser.add_argument('--source_dataset', type=str, default='DK', help='dataset name: DK')
    parser.add_argument('--infer_dataset', type=str, default='SP', help='dataset name: SP, slovenia, slovakia')
    parser.add_argument('--target', type=str, default='height', help='target name: height, count, treecover5m')
    parser.add_argument('--backbone', type=str, default='vit_b16_reduce', help='backbone name: vit_b16_reduce, vgg16v2norm_reduce')
    parser.add_argument('--few_shot_num', type=int, default=5, help='few shot number')
    args = parser.parse_args()

    cfg = Config(args.source_dataset, args.infer_dataset, args.target, args.backbone, args.few_shot_num)
    main(cfg)
