import os
import logging
import shutil
import time
import types
import argparse
from typing import Tuple
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, random_split

from utils import EarlyStopper
from data import Movilens1MDataSet
from model import  NMF, NMFSlate, NMFPfd, WD, WDSlate, WDPfd, DeepFM, DeepFMslate, DeepFMPfd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='movilens1M')
    parser.add_argument('--dataset_path', type=str, default='./datahub/movielens1M/movilens1M_slate_data.pkl')
    parser.add_argument('--model_name', type=str, default='nmfslate', help="[nmf, nmfslate, nmfpfd, wd, wdslate, wdpfd, deepfm, deepfmslate, deepfmpfd]")
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bsz', type=int, default=2048)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_dir', type=str, default='chkpt')
    args = parser.parse_args()
    return args

def get_logger(args):
    localtime = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, "{}-{}".format(localtime, args.model_name))
    args.save_dir = save_dir
    if os.path.exists(save_dir):
        shutil.rmdir(save_dir)
    os.makedirs(save_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(os.path.join(save_dir, "output.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("{} Args Setting {}".format('*'*30, '*'*30))
    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))
    return save_dir, logger

def get_loaders(args):
    if args.dataset_name == 'movilens1M':
        feature_config = {"user_id": 6041, "item_id": 3953, "slate_size": 19}
        dataset = Movilens1MDataSet(args.dataset_name, args.dataset_path, args.device)
    else:
        raise NotImplemented
    train_size, val_size = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, len(dataset)-train_size-val_size])
    train_dl = DataLoader(train_set, batch_size=args.bsz, shuffle=args.shuffle)
    val_dl = DataLoader(val_set, batch_size=args.bsz, shuffle=args.shuffle)
    test_dl = DataLoader(test_set, batch_size=args.bsz, shuffle=args.shuffle)
    return train_dl, val_dl, test_dl, feature_config

def train(model, optimizer, criterion, train_dl, log_interval=100, is_pfd=False):
    model.train()
    total_loss, a, b, c = 0, 0, 0 ,0
    tk0 = tqdm.tqdm(train_dl, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        users, items, slate_ids, slate_poses, slate_ratings = \
            fields['user_id'], fields['item_id'], fields['slate_id'], fields['slate_pos'], fields['slate_rating']
        y = model(users, items, slate_ids, slate_poses, slate_ratings)
        if is_pfd:
            output_t, output_s, reg_loss = y
            a += criterion(output_t, target.float()).item()
            b += criterion(output_s, target.float()).item()
            c += reg_loss.item()
            loss = criterion(output_t, target.float()) + criterion(output_s, target.float()) + reg_loss
        elif isinstance(y, Tuple):
            y, reg_loss = y
            loss = criterion(y, target.float()) + reg_loss
        else:
            loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            if is_pfd:
                tk0.set_postfix(loss=total_loss / log_interval, loss_t = a / log_interval, loss_s = b / log_interval, reg_term = c / log_interval)
                total_loss, a, b, c = 0, 0, 0 ,0
            else:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

def test(model, data_loader, is_pfd=0):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            users, items, slate_ids, slate_poses, slate_ratings = \
                fields['user_id'], fields['item_id'], fields['slate_id'], fields['slate_pos'], fields['slate_rating']
            y = model(users, items, slate_ids, slate_poses, slate_ratings)
            if is_pfd:
                _, y, _ = y
            elif isinstance(y, Tuple):
                y, _ = y
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def get_model(args, feature_config):
    if args.model_name.lower() == 'nmf':
        return NMF(args.emb_dim, feature_config, mlp_dims=(16, 16), out_alpha=0.5)
    elif args.model_name.lower() == 'nmfslate':
        return NMFSlate(args.emb_dim, feature_config, mlp_dims=(16, 16), out_alpha=0.5)
    elif args.model_name.lower() == 'nmfpfd':
        return NMFPfd(args.emb_dim, feature_config, mlp_dims=(16, 16), out_alpha=0.5)
    elif args.model_name.lower() == 'wd':
        return WD(args.emb_dim, feature_config, mlp_dims=(16, 16))
    elif args.model_name.lower() == 'wdslate':
        return WDSlate(args.emb_dim, feature_config, mlp_dims=(16, 16))
    elif args.model_name.lower() == 'wdpfd':
        return WDPfd(args.emb_dim, feature_config, mlp_dims=(16, 16))
    else:
        raise NotImplemented

if __name__ == '__main__':
    args = get_args()
    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    save_dir, logger = get_logger(args) 
    train_dl, val_dl, test_dl, feature_config = get_loaders(args)
    model = get_model(args, feature_config).to(args.device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{args.model_name}.pt')
    logger.info("{} Training Monitor {}".format('*'*30, '*'*30))
    print(args.model_name)
    for epoch_i in range(args.epoch):
        train(model, optimizer, criterion, train_dl, is_pfd=('pfd' in args.model_name.lower()))
        auc = test(model, val_dl, is_pfd=('pfd' in args.model_name.lower()))
        logger.info('epoch: {:d} validation auc: {:.5f}'.format(epoch_i, auc))
        if not early_stopper.is_continuable(model, auc):
            logger.info('validation: best auc: {:.5f}'.format(early_stopper.best_accuracy))
            break
    auc = test(model, test_dl, is_pfd=('pfd' in args.model_name.lower()))
    logger.info('test auc: {:.5f}'.format(auc))


        

