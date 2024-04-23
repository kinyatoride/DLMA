import os
import torch
import numpy as np
from loss import ma_loss, ma_fcst_err

def test(
    model, device, dataloader, 
    t0_library, t0_mask, n_sub, t1_library,
    n_analog, insample=False,
):    
    model.eval()
    test_loss = []
    test_mean_mse = []
    test_mse = []
    t0_indices = []
    weights = []
    data_size = len(dataloader.dataset)

    with torch.no_grad():
        for x0, x1, t1_dist in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t1_dist = t1_dist.to(device)
            batch_size = x0.shape[0]
            
            weight = model(x0)
            weight = torch.where(t0_mask[None], 0, weight)
            
            loss, mean_mse, indices = ma_loss(
                x0, t0_library, weight, t1_dist, n_sub, 
                insample=insample, return_index=True)
            
            mse = ma_fcst_err(
                x1, t1_library, indices, n_analog=n_analog,
            )
            
            test_loss.append(loss)
            test_mean_mse.append(mean_mse * batch_size)
            test_mse.append(mse * batch_size)
            t0_indices.append(indices)
            weights.append(weight)

    test_loss = torch.tensor(test_loss).mean().item()
    test_mean_mse = torch.tensor(test_mean_mse).sum().item() / data_size
    test_mse = torch.tensor(test_mse).sum().item() / data_size
    t0_indices = torch.cat(t0_indices).cpu().detach().numpy()
    weights = torch.cat(weights).cpu().detach().numpy()
    
    return test_loss, test_mean_mse, test_mse, t0_indices, weights

def test_uniform(
    model, device, dataloader, 
    t0_library, t0_mask, n_sub, t1_library,
    n_analog, insample=False,
):    
    model.eval()
    test_loss = []
    test_mean_mse = []
    test_mse = []

    data_size = len(dataloader.dataset)

    with torch.no_grad():
        for x0, x1, t1_dist in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t1_dist = t1_dist.to(device)
            batch_size = x0.shape[0]
            
            weight = model(x0)
            weight = weight * 0 + 1   # Uniform weights
            weight = torch.where(t0_mask[None], 0, weight)
            
            loss, mean_mse, indices = ma_loss(
                x0, t0_library, weight, t1_dist, n_sub, 
                insample=insample, return_index=True)
            
            mse = ma_fcst_err(
                x1, t1_library, indices, n_analog=n_analog,
            )
            
            test_loss.append(loss)
            test_mean_mse.append(mean_mse * batch_size)
            test_mse.append(mse * batch_size)

    test_loss = torch.tensor(test_loss).mean().item()
    test_mean_mse = torch.tensor(test_mean_mse).sum().item() / data_size
    test_mse = torch.tensor(test_mse).sum().item() / data_size
    
    return test_loss, test_mean_mse, test_mse


def test_n_analogs(
    model, device, dataloader, 
    t0_library, t0_mask, n_sub, t1_library,
    n_analogs,
    insample=False,
):    
    model.eval()
    test_mse = []
    data_size = len(dataloader.dataset)

    with torch.no_grad():
        for x0, x1, t1_dist in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            t1_dist = t1_dist.to(device)
            batch_size = x0.shape[0]
            
            weight = model(x0)
            weight = torch.where(t0_mask[None], 0, weight)
            
            loss, mean_mse, indices = ma_loss(
                x0, t0_library, weight, t1_dist, n_sub, 
                insample=insample, return_index=True)
            
            lst = []
            for n_analog in n_analogs:
                mse = ma_fcst_err(
                    x1, t1_library, indices, n_analog=n_analog,
                )
                lst.append(mse * batch_size)
            
            test_mse.append(torch.tensor(lst))

    mse = torch.stack(test_mse).sum(axis=0).cpu().detach().numpy() / data_size
    
    return mse