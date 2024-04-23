"""
Modules for evaluating significance
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
from utils import to_monthly, shift_time, extract_month_and_to_yearly

def rank_to_pval(rank, side='two-sided'):
    """
    Convert percentile rank to p-value

    Parameters
    ----------
    rank : float or numpy.ndarray
        percentile ranks ranging [0, 1]
    side : {'two-sided', 'less', 'greater'}, optional
    
    Returns
    -------
    pval : float or numpy.ndarray
        p-values
    """
    if side == 'two-sided':
        pval = rank
        if isinstance(rank, float):
            if pval > 0.5:
                pval = 1 - pval
        else:
            pval[pval > 0.5] = (1 - pval[pval > 0.5])
        pval *= 2

    elif side == 'less':
        pval = rank
    elif side == 'greater':
        pval = 1 - rank
    else:
        raise ValueError("side must be either 'two-sided', 'less', or 'greater'.")
        
    return pval 

def quantileofscore(a, score, dim=None):
    """
    Similar to scipy.stats.percentileofscore but accepts dim and nan values.

    Parameters
    ----------
    a : tenso (n, ...)
        Array to which score is compared.
    score : float or tensor (...)
        Scores to compute percentiles for.
    dim : None or int
        dim along which quantiles are computed.
    """

    if dim is not None:
        score = torch.unsqueeze(score, dim=dim)

    # Count lower values
    n_low = (a < score).sum(dim=dim) 

    # Count same values
    n_same = (a == score).sum(dim=dim).to(torch.float)

    # Count finite values
    n_all = torch.isfinite(a).sum(dim=dim)

    # Find median
    if not torch.is_tensor(score):
        if n_same > 0:
            n_same = (n_same + 1) / 2

        # quantile
        if n_all != 0:
            q = (n_low + n_same) / n_all
        else:
            q = np.nan
    else:
        n_same[n_same > 0] = (n_same[n_same > 0] + 1) / 2

        # quantile
        q = (n_low + n_same) / n_all
        q[n_all == 0] = np.nan

    return q

def mean_func(x, dim=None, weights=None):
    """
    Parameters
    ----------
    x : (sample, ..., n, ..., m)
    weights : (n, ..., m)
        The dimensions of weights should match with the last dimensions of x
        If weights are provided, dim is ignored
    """
    if weights is None:
        # print('weights is None')
        return x.nanmean(dim=dim)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            # Mask uac
            x_ma = torch.masked.masked_tensor(x, torch.isfinite(x))

            # Mask weight
            new_shape = tuple(1 for _ in x.shape[:x.dim() - weights.dim()]) + weights.shape
            weights_ex = weights.view(new_shape).expand(x.shape)
            weights_ma = torch.masked.masked_tensor(weights_ex, torch.isfinite(x))

            dim = list(x.dim() - weights.dim() + torch.arange(weights.dim()))
            x_mean = (x_ma * weights_ma).sum(dim=dim) / weights_ma.sum(dim=dim)

            return x_mean.to_tensor(np.nan)

def root_mean_func(x, dim=None, weights=None):
    return mean_func(x, dim=dim, weights=weights) ** 0.5

def uac_func(x, y, dim=None):
    return ((x * y).sum(dim=dim) 
            / ((x**2).sum(dim=dim)
               * (y**2).sum(dim=dim)) ** 0.5)

def uac2_func(x, y, dim=None):
    """Squared anomaly correlation"""
    uac = uac_func(x, y, dim=dim)
    uac[uac < 0] = 0   # Remove negative correlation
    return uac**2

def mse_func(x, y, dim=None, weights=None):
    return mean_func((x - y)**2, dim=dim, weights=weights)

def rmse_func(x, y, dim=None, weights=None):
    return mean_func((x - y)**2, dim=dim, weights=weights) ** 0.5

def mean_uac_func(x, y, dim=None, weights=None, mean_dim=None):
    return mean_func(uac_func(x, y, dim=dim), dim=mean_dim, weights=weights)

def mean_uac2_func(x, y, dim=None, weights=None, mean_dim=None):
    return mean_func(uac2_func(x, y, dim=dim), dim=mean_dim, weights=weights)

def mean_mse_func(x, y, dim=None, weights=None, mean_dim=None):
    return mean_func(mse_func(x, y, dim=dim), dim=mean_dim, weights=weights)

def get_data_lead(da, x, y, lead, month=None):
    """
    Prepare aligned data for a specific lead
    """
    # Get reference
    da_shift = shift_time(da, lead)

    if month is None:
        ref = to_monthly(da_shift)
    else:
        ref = extract_month_and_to_yearly(da_shift, month)

    # Align dimension size
    ref, x_lead, y_lead = xr.align(ref, x.sel(lead=lead), y.sel(lead=lead))

    return ref, x_lead, y_lead

def permute_stat_diff(
        stat_func, x, y, dim, 
        n_resamples=1000, batch=100,
        alternative='greater', device='cpu',
        out_dim=None, return_null=False,
        **stat_kw,
    ):
    """
    Permute paired samples and perform a statistical test on the difference of a given statistics.
    Similar to scipy.stats.permutation_test with permutation_type='samples' but with GPUs.
    
    Difference in statistics is defined as: stat_func(x) - stat_func(y)

    Parameters
    ----------
    stat_func : function
        Statistic function that accepts (pred, dim)
    x : xarray, (lead, year, ...)
        First set of predicted values
    y : xarray, (lead, year, ...)
        Second set of predicted values
    dim : list of str
        Dimensions along which the summary statistic is calculated
    n_resamples : int
        Number of random permutations (resamples) 
    batch : int
        The number of permutations to process in each call to stat_func.
    alternative : {‘two-sided’, ‘less’, ‘greater’}
        The alternative hypothesis for which the p-value is calculated.
    device : str

    out_dim : list of str

    return_null : bool 
        If true, return the null distribution
        
    Returns
    -------
    res_ds : xarray.Dataset
        Permutation test result
        stat : 
            The observed test statistic.
        pval : 
            The p-value of the given alternative.
    """
    lst = []
    lst_null = []
    for lead in x.lead.data:
        # print(f'{lead}-month lead')

        # Align dimension size
        x_lead_da, y_lead_da = xr.align(x.sel(lead=lead), y.sel(lead=lead))

        # Reshape, transpose, and convert to tensor
        x_lead = torch.from_numpy(x_lead_da.stack(sample=dim).transpose('sample', ...).data).to(device)
        y_lead = torch.from_numpy(y_lead_da.stack(sample=dim).transpose('sample', ...).data).to(device)
        n_samples = x_lead.shape[0]

        # stat
        x_stat = stat_func(x_lead, dim=0, **stat_kw)
        y_stat = stat_func(y_lead, dim=0, **stat_kw)
        stat = x_stat - y_stat 

        # Combine two samples
        xy = torch.stack((x_lead, y_lead), dim=0)

        # Create random indices
        idx_shape = (n_resamples, n_samples) + tuple(1 for _ in x_lead.shape[1:])
        idx1 = torch.randint(2, size=idx_shape, device=device)
        idx2 = 1 - idx1
        # print(f'{idx1.shape = :}')

        # Expand
        idx1_ex = idx1.expand((n_resamples, n_samples) + tuple(x_lead.shape[1:]))
        idx2_ex = idx2.expand((n_resamples, n_samples) + tuple(x_lead.shape[1:]))
        # print(f'{idx1_ex.shape = :}')

        if batch is None:
            batch = n_resamples

        # Split indices
        idx1_batchs = torch.split(idx1_ex, batch, dim=0)
        idx2_batchs = torch.split(idx2_ex, batch, dim=0)

        null_dist = []
        for idx1_batch, idx2_batch in zip(idx1_batchs, idx2_batchs):
            # Samples of statistic
            x_batch = torch.gather(xy, 0, idx1_batch)
            y_batch = torch.gather(xy, 0, idx2_batch)
            #print(f'{x_batch.shape = :}')

            # Samples of statistic
            x_stat_batch = stat_func(x_batch, dim=1, **stat_kw)
            y_stat_batch = stat_func(y_batch, dim=1, **stat_kw)
            #print(f'{x_stat_batch.shape = :}')

            # stat_diff
            null_dist.append(x_stat_batch - y_stat_batch)
            
        null_dist = torch.cat(null_dist, dim=0)

        # Compute the quantile rank of a score 
        q = quantileofscore(null_dist, stat, dim=0)

        # pvalue
        pval = rank_to_pval(q, alternative)

        # To xarray
        if out_dim is None:
            out_dim = [key for key in x_lead_da.dims if key not in dim]
            
        res_ds = xr.Dataset(
            {'stat': (out_dim, stat.cpu().detach().numpy()),
             'pval': (out_dim, pval.cpu().detach().numpy()),
             'x_stat': (out_dim, x_stat.cpu().detach().numpy()),
             'y_stat': (out_dim, y_stat.cpu().detach().numpy()),
            },
            coords={key: x_lead_da.coords[key] for key in out_dim}
        )
        lst.append(res_ds)

        if return_null:
            null_dist = xr.DataArray(
                null_dist.cpu().detach().numpy(), 
                dims=['resample'] + out_dim,
                coords={key: x_lead_da.coords[key] for key in out_dim},
                name='null',
            )
            
            lst_null.append(null_dist)

    res_ds = xr.concat(lst, x.lead)
    
    if return_null:
        null_dist = xr.concat(lst_null, x.lead)
        return res_ds, null_dist
    else:
        return res_ds

def permute_statref_diff(
        stat_func, da, x, y, dim, month=None, 
        n_resamples=1000, batch=100,
        alternative='greater', device='cpu',
        out_dim=None, return_null=False,
        **stat_kw,
    ):
    """
    Permute paired samples and perform a statistical test on the difference 
    of a given statistics that requires reference data.
    Similar to scipy.stats.permutation_test with permutation_type='samples' but with GPUs.
    
    Difference in statistics is defined as: stat_func(x, ref) - stat_func(y, ref)

    Parameters
    ----------
    stat_func : function
        Statistic function that accepts (ref, pred, dim)
    da : xarray (time, ...)
        Reference dataset
    x : xarray, (lead, year, ...)
        First set of predicted values
    y : xarray, (lead, year, ...)
        Second set of predicted values
    dim : list of str
        Dimensions along which the summary statistic is calculated
    month : int or None
        Month of data
    n_resamples : int
        Number of random permutations (resamples) 
    batch : int
        The number of permutations to process in each call to stat_func.
    alternative : {‘two-sided’, ‘less’, ‘greater’}
        The alternative hypothesis for which the p-value is calculated.
    device : str

    out_dim : list of str

    return_null : bool 
        If true, return the null distribution
        
    Returns
    -------
    res_ds : xarray.Dataset
        Permutation test result
        stat : 
            The observed test statistic.
        pval : 
            The p-value of the given alternative.
    """
    lst = []
    lst_null = []
    for lead in x.lead.data:
        # print(f'{lead}-month lead')

        # Get aligned data
        ref_da, x_lead, y_lead = get_data_lead(da, x, y, lead, month=month)

        # Reshape, transpose, and convert to tensor
        ref = torch.from_numpy(ref_da.stack(sample=dim).transpose('sample', ...).data).to(device)
        x_lead = torch.from_numpy(x_lead.stack(sample=dim).transpose('sample', ...).data).to(device)
        y_lead = torch.from_numpy(y_lead.stack(sample=dim).transpose('sample', ...).data).to(device)
        n_samples = ref.shape[0]
        # print(f'{ref.shape = :}')
        # print(f'{x_lead.shape = :}')

        # stat
        x_stat = stat_func(ref, x_lead, dim=0, **stat_kw)
        y_stat = stat_func(ref, y_lead, dim=0, **stat_kw)
        stat = x_stat - y_stat 
        # print(f'{x_stat.shape = :}')

        # Combine two samples
        xy = torch.stack((x_lead, y_lead), dim=1)

        # Create random indices
        idx_shape = (n_samples, n_resamples) + tuple(1 for _ in ref.shape[1:])
        idx1 = torch.randint(2, size=idx_shape, device=device)
        idx2 = 1 - idx1
        # print(f'{idx1.shape = :}')

        # Expand
        idx1_ex = idx1.expand((n_samples, n_resamples) + tuple(ref.shape[1:]))
        idx2_ex = idx2.expand((n_samples, n_resamples) + tuple(ref.shape[1:]))
        # print(f'{idx1_ex.shape = :}')

        if batch is None:
            batch = n_resamples

        # Split indices
        idx1_batchs = torch.split(idx1_ex, batch, dim=1)
        idx2_batchs = torch.split(idx2_ex, batch, dim=1)

        null_dist = []
        for idx1_batch, idx2_batch in zip(idx1_batchs, idx2_batchs):
            # Samples of statistic
            x_batch = torch.gather(xy, 1, idx1_batch)
            y_batch = torch.gather(xy, 1, idx2_batch)
            # print(f'{x_batch.shape = :}')

            # Samples of statistic
            x_stat_batch = stat_func(torch.unsqueeze(ref, dim=1), x_batch, dim=0, **stat_kw)
            y_stat_batch = stat_func(torch.unsqueeze(ref, dim=1), y_batch, dim=0, **stat_kw)

            # stat_diff
            null_dist.append(x_stat_batch - y_stat_batch)
            
        null_dist = torch.cat(null_dist, dim=0)

        # Compute the quantile rank of a score 
        q = quantileofscore(null_dist, stat, dim=0)

        # pvalue
        pval = rank_to_pval(q, alternative)

        # To xarray
        if out_dim is None:
            out_dim = [key for key in ref_da.dims if key not in dim]
            
        res_ds = xr.Dataset(
            {'stat': (out_dim, stat.cpu().detach().numpy()),
             'pval': (out_dim, pval.cpu().detach().numpy()),
             'x_stat': (out_dim, x_stat.cpu().detach().numpy()),
             'y_stat': (out_dim, y_stat.cpu().detach().numpy()),
            },
            coords={key: ref_da.coords[key] for key in out_dim}
        )
        lst.append(res_ds)

        if return_null:
            null_dist = xr.DataArray(
                null_dist.cpu().detach().numpy(), 
                dims=['resample'] + out_dim,
                coords={key: ref_da.coords[key] for key in out_dim},
                name='null',
            )
            
            lst_null.append(null_dist)

    res_ds = xr.concat(lst, x.lead)
    
    if return_null:
        null_dist = xr.concat(lst_null, x.lead)
        return res_ds, null_dist
    else:
        return res_ds

# def permute_statref_ndiff(
#         stat_func, da, x, y, dim, month=None, 
#         n_resamples=1000, batch=100,
#         alternative='greater', device='cpu',
#         out_dim=None, return_null=False,
#         **stat_kw,
#     ):
#     """
#     Permute paired samples and perform a statistical test on the difference 
#     of a given statistics that requires reference data.
#     Similar to scipy.stats.permutation_test with permutation_type='samples' but with GPUs.
    
#     Difference in statistics is defined as: stat_func(x, ref) - stat_func(y, ref)

#     Parameters
#     ----------
#     stat_func : function
#         Statistic function that accepts (ref, pred, dim)
#     da : xarray (time, ...)
#         Reference dataset
#     x : xarray, (lead, year, ...)
#         First set of predicted values
#     y : xarray, (lead, year, ...)
#         Second set of predicted values
#     dim : str or list of str
#         Dimensions along which the summary statistic is calculated
#     month : int or None
#         Month of data
#     n_resamples : int
#         Number of random permutations (resamples) 
#     batch : int
#         The number of permutations to process in each call to stat_func.
#     alternative : {‘two-sided’, ‘less’, ‘greater’}
#         The alternative hypothesis for which the p-value is calculated.
#     device : str

#     out_dim : list of str

#     return_null : bool 
#         If true, return the null distribution
        
#     Returns
#     -------
#     res_ds : xarray.Dataset
#         Permutation test result
#         stat : 
#             The observed test statistic.
#         pval : 
#             The p-value of the given alternative.
#     """
#     lst = []
#     lst_null = []
#     for lead in x.lead.data:
#         print(f'{lead}-month lead')

#         # Get aligned data
#         ref_da, x_lead, y_lead = get_data_lead(da, x, y, lead, month=month)

#         # Reshape, transpose, and convert to tensor
#         ref = torch.from_numpy(ref_da.stack(sample=dim).transpose('sample', ...).data).to(device)
#         x_lead = torch.from_numpy(x_lead.stack(sample=dim).transpose('sample', ...).data).to(device)
#         y_lead = torch.from_numpy(y_lead.stack(sample=dim).transpose('sample', ...).data).to(device)
#         n_samples = ref.shape[0]

#         # stat
#         x_stat = stat_func(ref, x_lead, dim=0, **stat_kw)
#         y_stat = stat_func(ref, y_lead, dim=0, **stat_kw)
#         stat = (x_stat - y_stat) / x_stat

#         # Combine two samples
#         xy = torch.stack((x_lead, y_lead), dim=1)

#         # Create random indices
#         idx_shape = (n_samples, n_resamples) + tuple(1 for _ in ref.shape[1:])
#         idx1 = torch.randint(2, size=idx_shape, device=device)
#         idx2 = 1 - idx1

#         # Expand
#         idx1_ex = idx1.expand((n_samples, n_resamples) + tuple(ref.shape[1:]))
#         idx2_ex = idx2.expand((n_samples, n_resamples) + tuple(ref.shape[1:]))

#         if batch is None:
#             batch = n_resamples

#         # Split indices
#         idx1_batchs = torch.split(idx1_ex, batch, dim=1)
#         idx2_batchs = torch.split(idx2_ex, batch, dim=1)

#         null_dist = []
#         for idx1_batch, idx2_batch in zip(idx1_batchs, idx2_batchs):
#             # Samples of statistic
#             x_batch = torch.gather(xy, 1, idx1_batch)
#             y_batch = torch.gather(xy, 1, idx2_batch)

#             # Samples of statistic
#             x_stat_batch = stat_func(torch.unsqueeze(ref, dim=1), x_batch, dim=0, **stat_kw)
#             y_stat_batch = stat_func(torch.unsqueeze(ref, dim=1), y_batch, dim=0, **stat_kw)

#             # stat_diff
#             null_dist.append((x_stat_batch - y_stat_batch) / x_stat_batch)
            
#         null_dist = torch.cat(null_dist, dim=0)

#         # Compute the quantile rank of a score 
#         q = quantileofscore(null_dist, stat, dim=0)

#         # pvalue
#         pval = rank_to_pval(q, alternative)

#         # To xarray
#         if out_dim is None:
#             out_dim = [key for key in ref_da.dims if key not in dim]
            
#         res_ds = xr.Dataset(
#             {'stat': (out_dim, stat.cpu().detach().numpy()),
#              'pval': (out_dim, pval.cpu().detach().numpy()),
#             },
#             coords={key: ref_da.coords[key] for key in out_dim}
#         )
#         lst.append(res_ds)

#         if return_null:
#             null_dist = xr.DataArray(
#                 null_dist.cpu().detach().numpy(), 
#                 dims=['resample'] + out_dim,
#                 coords={key: ref_da.coords[key] for key in out_dim},
#             )
            
#             lst_null.append(null_dist)

#     res_ds = xr.concat(lst, x.lead)
    
#     if return_null:
#         null_dist = xr.concat(lst_null, x.lead)
#         return res_ds, null_dist
#     else:
#         return res_ds

#     #     if batch is None:
#     #         # Create random indices
#     #         idx_shape = (n_samples, n_resamples) + tuple(ref.shape[1:])
#     #         idx1 = torch.randint(2, size=idx_shape, device=device)
#     #         idx2 = 1 - idx1
        
#     #         # Resample
#     #         x_samples = torch.gather(xy, 1, idx1)
#     #         y_samples = torch.gather(xy, 1, idx2)

#     #         # Samples of statistic
#     #         x_stat_samples = stat_func(torch.unsqueeze(ref, dim=1), x_samples, dim=0, **stat_kw)
#     #         y_stat_samples = stat_func(torch.unsqueeze(ref, dim=1), y_samples, dim=0, **stat_kw)

#     #         # stat diff
#     #         null_dist = x_stat_samples - y_stat_samples
#     #     else:
#     #         resample_idxs = torch.split(torch.arange(n_resamples), batch)
#     #         null_dist = []
#     #         for resample_idx in resample_idxs:

#     #             # Create random indices
#     #             idx_shape = (n_samples, len(resample_idx)) + tuple(ref.shape[1:])
#     #             idx1 = torch.randint(2, size=idx_shape, device=device)
#     #             idx2 = 1 - idx1

#     #             # Samples of statistic
#     #             x_batch = torch.gather(xy, 1, idx1)
#     #             y_batch = torch.gather(xy, 1, idx2)