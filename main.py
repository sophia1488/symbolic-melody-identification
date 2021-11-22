import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import tqdm
import os
from pathlib import Path
import json
import random

from model import CNN_Net
from dataset import PianorollDataset
from preprocess import preprocess, data_augmentation
from train import training, valid
import utils


def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return


def get_args():
    parser = argparse.ArgumentParser()
    ### path ###
    parser.add_argument('--split_file', type=str, default='pop909_datasplit.pkl')
    parser.add_argument('--root', type=str, default='../Dataset/pop909_aligned', help='path to pop909 dataset')
    parser.add_argument('--name', type=str, required=True, help='path where training result will be saved')
    
    ### parameter ###
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--train_batch', type=int, default=96)
    parser.add_argument('--valid_batch', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--data_augment', type=bool, default=True)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # fix seed
    fix_seed(args.seed)

    # set path
    exp_dir = f'result/{args.name}'
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # prepare data
    train_score, train_melody, valid_score, valid_melody = preprocess(args.split_file, args.root) 
    if args.data_augment:
        X, y = data_augmentation(train_score, train_melody)
        train_score = np.concatenate((train_score, X))
        train_melody = np.concatenate((train_melody, y))
    
    print(f'train shape: {train_score.shape}, {train_melody.shape}')    # (67117, 1, 128, 64)
    print(f'valid shape: {valid_score.shape}, {valid_melody.shape}\n')  # (8295, 1, 128, 64)
   
    # dataloader
    trainset = PianorollDataset(X=train_score, y=train_melody)
    validset = PianorollDataset(X=valid_score, y=valid_melody)
    train_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True)
    print(f'len of train loader: {len(train_loader)}')
    valid_loader = DataLoader(validset, batch_size=args.valid_batch)
    print(f'len of valid loader: {len(valid_loader)}')

    # model
    print(f'\nInitializing CNN...')
    model = CNN_Net().to(device)
    model.init_weights()

    lr = args.lr
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = lr,
        weight_decay = 0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=80,
        cooldown=10
    )
    es = utils.EarlyStopping(patience = args.patience)
    
    # train
    t = tqdm.trange(1, args.epoch+1, disable=False)
    train_losses, valid_losses = [], []
    best_epoch, stop_t = 0, 0
    
    print(f'Start training')

    for epoch in t:
        t.set_description("Training Epoch")
        train_loss = training(model, device, train_loader, optimizer)
        valid_loss = valid(model, device, valid_loader)

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)
        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best = valid_loss==es.best,
            path = exp_dir,
            target = 'CNN-melody-identification'
        )

        params = {
            'epochs_trained': epoch,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'num_bad_epochs': es.num_bad_epochs,
        }

        with open(os.path.join(exp_dir, 'CNN-melody.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))       

        if stop:
            print("Apply Early Stopping and Retrain")
            stop_t += 1
            if stop_t >= 5: break
            lr *= 0.5
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr = lr,
                weight_decay = 0.01
            )
            es = utils.EarlyStopping(patience = args.patience, best_loss = es.best)


if __name__ == '__main__':
    main()
