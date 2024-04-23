"""
Modules for evaluating significance
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import permutation_test
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

def quantileofscore(a, score, axis=None):
    """
    Similar to scipy.stats.percentileofscore but accepts axis and nan values.

    Parameters
    ----------
    a : numpy.ndarray (n, ...)
        Array to which score is compared.
    score : float or numpy.ndarray (...)
        Scores to compute percentiles for.
    axis : None or int
        Axis along which quantiles are computed.
    """

    if axis is not None:
        score = np.expand_dims(score, axis=axis)

    # Count lower values
    n_low = (a < score).sum(axis=axis) 

    # Count same values
    n_same = (a == score).sum(axis=axis)

    # Count finite values
    n_all = np.isfinite(a).sum(axis=axis)

    # Find median
    if not isinstance(score, np.ndarray):
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
        nan_array = np.empty(n_all.shape)
        nan_array[:] = np.nan
        q = np.divide(n_low + n_same, n_all, out=nan_array, where=n_all!=0)

    return q

def permute_acu_diff(da, x, y, dim, month=None, 
                     n_resamples=1000, batch=100,
                     alternative='greater'):
    """
    Permute ACU difference = ACU_x - ACU_y

    Parameters
    ----------
    da : xarray.dataarray (time, ...)
        Reference data
    x : xarray.dataarray, (lead, year, ...)
        The predicted value 
    y : xarray.dataarray, (lead, year, ...)
        The predicted value (control)
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    n_resamples, batch, alternative
        Parameters passed to `scipy.stats.permutation_test`

    Returns
    -------
    acu_diff : xarray.dataset (lead, year, ...)
        stat : acu difference
        pval : p-values
    """

    lst = []
    for lead in x.lead.data:
        print(f'ACU diff: {lead}-month lead')

        # Get reference
        da_shift = shift_time(da, lead)

        if month is None:
            ref_da = to_monthly(da_shift)
        else:
            ref_da = extract_month_and_to_yearly(da_shift, month)

        # Align dimension size
        ref_da, x_lead, y_lead = xr.align(ref_da, x.sel(lead=lead), y.sel(lead=lead))

        # Reshape and transpose
        ref = ref_da.stack(sample=dim).transpose(..., 'sample').data
        x_lead = x_lead.stack(sample=dim).transpose(..., 'sample').data
        y_lead = y_lead.stack(sample=dim).transpose(..., 'sample').data

        # Define a function here (ref is updated every lead)
        def eval_acu_diff(pred1, pred2, axis):
            acu1 = ((ref * pred1).sum(axis=axis) 
                / ((ref**2).sum(axis=axis) 
                    * (pred1**2).sum(axis=axis)) ** 0.5) 
            acu2 = ((ref * pred2).sum(axis=axis) 
                / ((ref**2).sum(axis=axis) 
                    * (pred2**2).sum(axis=axis)) ** 0.5) 
            return acu1 - acu2

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            res = permutation_test(
                (x_lead, y_lead), eval_acu_diff, 
                permutation_type='samples', vectorized=True,
                n_resamples=n_resamples, batch=batch, 
                alternative=alternative, axis=-1,
            )

        # To xarray
        out_dim = [key for key in ref_da.dims if key not in dim]
        res_ds = xr.Dataset(
            {'stat': (out_dim, res.statistic),
             'pval': (out_dim, res.pvalue),
            },
            coords={key: ref_da.coords[key] for key in out_dim}
        )
        
        lst.append(res_ds)
    acu_diff = xr.concat(lst, x.lead)

    # mask nan
    mask = ref_da.stack(sample=dim).isel(sample=0, drop=True).notnull()
    acu_diff['pval'] = acu_diff['pval'].where(mask)

    return acu_diff

def eval_nrmse(diff1, diff2, axis):
    rmse1 = (diff1 ** 2).mean(axis=axis) ** 0.5
    rmse2 = (diff2 ** 2).mean(axis=axis) ** 0.5
    return (1 - rmse1/rmse2) * 100 

def permute_nrmse(da, x, y, dim, month=None, 
                  n_resamples=1000, batch=100,
                  alternative='greater'):
    """
    Permute RMSE skill = 1 - RMSE/RMSE_CTRL

    Parameters
    ----------
    da : xarray.dataarray (time, ...)
        Reference data
    x : xarray.dataarray, (lead, year, ...)
        The predicted value 
    y : xarray.dataarray, (lead, year, ...)
        The predicted value (control)
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    n_resamples, batch, alternative
        Parameters passed to `scipy.stats.permutation_test`

    Returns
    -------
    nrmse : xarray.dataset (lead, year, ...)
        stat : RMSE skill
        pval : p-values
    """

    lst = []
    for lead in x.lead.data:
        print(f'NRMSE: {lead}-month lead')

        # Get reference
        da_shift = shift_time(da, lead)

        if month is None:
            ref_da = to_monthly(da_shift)
        else:
            ref_da = extract_month_and_to_yearly(da_shift, month)

        # Align dimension size
        ref_da, x_lead, y_lead = xr.align(ref_da, x.sel(lead=lead), y.sel(lead=lead))

        # Reshape and transpose
        ref = ref_da.stack(sample=dim).transpose(..., 'sample').data
        x_lead = x_lead.stack(sample=dim).transpose(..., 'sample').data
        y_lead = y_lead.stack(sample=dim).transpose(..., 'sample').data

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            res = permutation_test(
                (x_lead - ref, y_lead - ref), eval_nrmse, 
                permutation_type='samples', vectorized=True,
                n_resamples=n_resamples, batch=batch, 
                alternative=alternative, axis=-1,
            )

        # To xarray
        out_dim = [key for key in ref_da.dims if key not in dim]
        res_ds = xr.Dataset(
            {'stat': (out_dim, res.statistic),
             'pval': (out_dim, res.pvalue),
            },
            coords={key: ref_da.coords[key] for key in out_dim}
        )
        
        lst.append(res_ds)
    nrmse = xr.concat(lst, x.lead)

    # mask nan
    mask = ref_da.stack(sample=dim).isel(sample=0, drop=True).notnull()
    nrmse['pval'] = nrmse['pval'].where(mask)

    return nrmse

def circular_block_bootstrap_indices(n_sample, block_size, n_resample, seed=None):
    """
    Return indices for circular block bootstrapping (with replacement)

    Parameters
    ----------
    n_sample : int
        Number of samples
    block_size : int 
        Block length
    n_resample : int
        Number of bootstrapping iterations
    seed : int, optional
        Seed to generate random numbers

    Returns
    -------
    idx : numpy.ndarray, (n_resample, n_sample)
        indices for bootstrapping
    """
    rng = np.random.default_rng(seed)

    n_block = np.ceil(n_sample / block_size).astype(int)

    # Starting indices
    start_idx = rng.integers(n_sample, size=(n_resample, n_block))

    # Repeat the starting indices and shift them
    idx = (np.repeat(start_idx, block_size, axis=1) 
           + np.tile(np.arange(block_size), (n_resample, n_block)))

    # Truncate the array to sample size, and estimate modulus for circular indices
    return idx[:, :n_sample] % n_sample

def circular_block_bootstrap(x, n_resample, block_size=None, batch=None, seed=None):
    """
    Circular block bootstrap. Return sample-mean resamples.

    Parameters
    ----------
    x : numpy.ndarray (n_sample,)
        Data to resample from. The first dimesion is used for resampling.
    n_resample : int
        Number of bootstrapping iterations
    block_size : int, optional 
        Block length. If not provided, use a recommended value
    batch : int, optional
        The number of resamples to process in each vectorized call
    seed : int, optional
        Seed to generate random numbers

    Returns
    -------
    sample_mean : numpy.ndarray (n_resample,)
    """

    n_sample = x.shape[0]
    if block_size is None:
        block_size = int(n_sample / n_sample**(1./3))

    # Get indices
    idx = circular_block_bootstrap_indices(n_sample, block_size, n_resample, seed=seed)

    if batch is None:
        sample_mean = x[idx].mean(axis=1)
    else:
        n_batch = int(np.ceil(n_resample / batch))
        idx_batchs = np.array_split(idx, n_batch)

        sample_mean = np.concatenate(
            [x[idx_batch].mean(axis=1) for idx_batch in idx_batchs]
        )
    
    return sample_mean