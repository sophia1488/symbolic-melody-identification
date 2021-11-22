import tqdm
import numpy as np
import torch
import torch.nn as nn


def weighted_bce_loss(pred, target, weight, eps=1e-15):
    assert len(weight) == 2
    loss = weight[0] * (target * torch.log(pred+eps)) +\
           weight[1] * ((1-target) * torch.log(1-pred+eps))
    return torch.neg(torch.mean(loss))


def training(model, device, data_loader, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(data_loader, disable=False)

    for x, y in pbar:
        pbar.set_description("Training Batch")
        x, y = x.to(device).float(), y.to(device).float()
        optimizer.zero_grad()
        
        # predict
        y_hat = model(x)          # (batch, 1, 64, 128)
        attn = (x!=0).float()     # (batch, 1, 64, 128) 

        # weight: most common / rare class
        # weight = torch.tensor([615039696/2735408, 1]) roughly 224.84
        weight = torch.tensor([1, 1])
        loss = weighted_bce_loss(y_hat, y, weight)

        loss = loss * attn
        loss = torch.sum(loss) / torch.sum(attn)
        
        # update
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss


def valid(model, device, data_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device).float(), y.to(device).float()

            # predict
            y_hat = model(x)

            attn = (x!=0).float()
            weight = torch.tensor([1, 1])
            loss = weighted_bce_loss(y_hat, y, weight)
            
            loss = loss * attn
            loss = torch.sum(loss) / torch.sum(attn)
            
            # update
            total_loss += loss.item()
        return total_loss
