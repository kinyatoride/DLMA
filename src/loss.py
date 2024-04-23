import torch
import torch.nn as nn
    
def max_scale(x, dim):
    return x / x.max(dim=dim, keepdim=True)[0] 

def ma_loss(x, library, weight, d1, n_sub=300, n_analog=30, 
            insample=True, return_index=False):
    """
    Return the loss of model analog forecasts
    
    Parameters
    ----------
    x : tensor, (batch, channel, lat, lon)
        Batch of initial conditions
    library : tensor, (sample, channel, lat, lon)
        Collection of samples
    weight : tensor, (batch, channel, lat, lon)
        Weights used to measure distance
    d1 : tensor, (batch, sample) 
        target distance
    n_sub : int
        Number of subsamples used to calculate loss
    n_analog : int
        Number of analogs
    insample : bool
        If library contains targert initial conditions
    return_index : bool
        If True, return analog indices
    
    Returns
    -------
    loss : tensor, (batch)
        Loss = (d0 - d1) ** 2 of subsamples
        d0: initial distance
        d1: target distance
    mean_mse : tensor, (batch)
        Mean of individual analog forecast errors
    t0_indices : tensor, (batch, sample) 
        indices used to select analogs
    """    
    # Weighted initial distance
    d0 = (((x[:, None] - library) ** 2 * weight[:, None]).sum(dim=[2, 3, 4]) 
          / weight[:, None].sum(dim=[2, 3, 4]))
    
    # Get indices for sorting
    t0_indices = torch.argsort(d0)
    
    if insample:
        # Remove the first indices
        t0_indices = t0_indices[:, 1:]  

    # Sort errors
    d0_sortby_d0 = torch.gather(d0, -1, t0_indices)
    d1_sortby_d0 = torch.gather(d1, -1, t0_indices)
        
    # Max-scale
    d0_nor = max_scale(d0_sortby_d0, 1)
    d1_nor = max_scale(d1_sortby_d0, 1)

    # L2 loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(d0_nor[:, :n_sub], d1_nor[:, :n_sub])
    
    # Mean of individual analog forecast errors
    mean_mse = d1_sortby_d0[:, :n_analog].mean()

    if return_index:    
        return loss, mean_mse, t0_indices
    else:
        return loss, mean_mse

def ma_fcst_err(x, library, indices, n_analog=30):
    """
    Return forecast error of analog-mean
    
    Parameters
    ----------
    x : tensor, (batch, lat, lon)
        Batch of target conditions
    library : tensor, (sample, lat, lon)
        Collection of samples at target time
    indices : tensor, (batch, sample) 
        indices used to select analogs    
    n_analog : int
        Number of analogs
    
    Returns
    -------
    mse : tensor, (batch)
        MSE of analog-mean forecast
    """
    
    # Analog-mean forecasts (batch, lat, lon)
    af = library[indices[:, :n_analog]].mean(dim=1)
    
    # mean-squared-error
    mse = ((x - af) ** 2).nanmean()
    
    return mse