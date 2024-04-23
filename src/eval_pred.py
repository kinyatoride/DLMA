"""
Modules for evaluating prediction
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from utils import extract_month_and_to_yearly, to_monthly, shift_time
xr.set_options(keep_attrs=True)

def eval_mse(x, y, dim):
    """
    Mean-squared-error
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(x.lat))  
        return ((x - y) ** 2).weighted(wgt).mean(dim=dim)   
    else:
        return ((x - y) ** 2).mean(dim=dim)

def eval_r(x, y, dim):
    """
    Pearson correlation
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(x.lat))  
        x_m = x.weighted(wgt).mean(dim=dim)
        y_m = y.weighted(wgt).mean(dim=dim)
        return (((x - x_m) * (y - y_m)).weighted(wgt).mean(dim=dim)
                / x.weighted(wgt).std(dim=dim)
                / y.weighted(wgt).std(dim=dim)
                )
    else:
        # Suppress runtime warning for empty array
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return xr.corr(x, y, dim=dim) 

def eval_uac(x, y, dim):
    """
    Uncentered anomaly correlation

    Parameters
    ----------
    x, y : xarray
        Anomalies
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(x.lat))  
        return ((x * y).weighted(wgt).sum(dim=dim)
                / ((x**2).weighted(wgt).sum(dim=dim) 
                   * (y**2).weighted(wgt).sum(dim=dim)) ** 0.5)
    else:
        return ((x * y).sum(dim=dim) 
                / ((x**2).sum(dim=dim) 
                   * (y**2).sum(dim=dim)) ** 0.5)
    
def eval_rmsss(y, y_hat, dim):
    """
    Root-mean-square skill score
    1 - rmse/rmse_clm

    Parameters
    ----------
    y : xarray
        The actual anomaly
    y_hat : xarray
        The predicted anomaly
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(y.lat))  
        rmse = ((y - y_hat) ** 2).weighted(wgt).mean(dim=dim) ** 0.5
        rmse_clm = (y ** 2).weighted(wgt).mean(dim=dim) ** 0.5
    else:
        rmse = ((y - y_hat) ** 2).mean(dim=dim) ** 0.5
        rmse_clm = (y ** 2).mean(dim=dim) ** 0.5
    return 1 - rmse / rmse_clm

def eval_msss(y, y_hat, dim):
    """
    Mean-square skill score (similar to NSE but for anomaly)
    1 - mse/mse_clm

    Parameters
    ----------
    y : xarray
        The actual anomaly
    y_hat : xarray
        The predicted anomaly
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(y.lat))  
        mse = ((y - y_hat) ** 2).weighted(wgt).mean(dim=dim)
        mse_clm = (y ** 2).weighted(wgt).mean(dim=dim)
    else:
        mse = ((y - y_hat) ** 2).mean(dim=dim)
        mse_clm = (y ** 2).mean(dim=dim) 
    return 1 - mse / mse_clm

def eval_nse(y, y_hat, dim):
    """
    Nash-Sutcliffe efficiency    
    1 - mse/sigma^2

    Parameters
    ----------
    y : xarray
        The actual value
    y_hat : xarray
        The predicted value
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(y.lat))  
        mse = ((y - y_hat) ** 2).weighted(wgt).mean(dim=dim)
        var = y.weighted(wgt).var(dim=dim)
    else:
        mse = ((y - y_hat) ** 2).mean(dim=dim)
        # Suppress runtime warning for empty array
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            var = y.var(dim=dim)
    return 1 - mse / var

def eval_kge(y, y_hat, dim):
    """
    Kling-Gupta efficiency

    Parameters
    ----------
    y : xarray
        The actual value
    y_hat : xarray
        The predicted value
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    """
    if 'lat' in dim:
        # grid weight
        wgt = np.cos(np.deg2rad(y.lat))  
        y_m = y.weighted(wgt).mean(dim=dim)
        y_hat_m = y_hat.weighted(wgt).mean(dim=dim)
        y_std = y.weighted(wgt).std(dim=dim)
        y_hat_std = y_hat.weighted(wgt).std(dim=dim)
    else:
        y_m = y.mean(dim=dim)
        y_hat_m = y_hat.mean(dim=dim)
        
        # Suppress runtime warning for empty array
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_std = y.std(dim=dim)
            y_hat_std = y_hat.std(dim=dim)

    r = eval_r(y, y_hat, dim)
    beta = y_hat_m / y_m
    gamma = y_hat_std / y_hat_m * y_m / y_std
    
    return 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5

def eval_bs(y, y_prob, dim='sample'):
    """
    Brier score

    Parameters
    ----------
    y : xarray.Dataarray (sample,)
        Occurances of the event (binary)
    y_prob : xarray.Dataarray (sample,)
        Predicted probabilities
    dim : str or list of str
        Dimensions along which the summary statistic is calculated

    Returns
    -------
    bs: xarray.Dataset
        Bier score, reliability, resolution
    y_prob_i: xarray.Dataarray
        Unique forecast values
    y_freq_i: xarray.Dataarray
        Conditional relative frequency
    n_i: xarray.Dataarray
        The number of each unique forecast value
    """  

    # Forecast values and their counts
    n_i = (y_prob).groupby(y_prob).count(dim=dim)
    y_prob_i = n_i.prob

    # Conditional relative frequency
    y_freq_i = y.groupby(y_prob).mean(dim=dim)

    # Overall relative frequency
    y_freq = y.mean(dim=dim)
        
    # Brier score
    bs = ((y_prob - y) ** 2).mean(dim=dim).rename('bs')
    bs.attrs['long_name'] = 'Brier score'

    # Reliability
    rel = ((y_prob_i - y_freq_i) ** 2).weighted(n_i).mean(dim='prob').rename('rel')
    rel.attrs['long_name'] = 'Reliability'

    # Resolution
    res = ((y_freq_i - y_freq) ** 2).weighted(n_i).mean(dim='prob').rename('res')
    res.attrs['long_name'] = 'Resolution'

    # Combine score
    bs = xr.merge([bs, rel, res])

    return bs, y_prob_i, y_freq_i, n_i

def eval_bs_tercile(y, y_hat, dim='sample', ensemble_dim='analog', return_prob=False):
    """
    Brier score for lower and upper terciles

    Parameters
    ----------
    y : xarray (sample,)
        The actual values
    y_hat : xarray (sample, ensemble,)
        The ensemble predicted values
    dim : str or list of str
        Dimensions along which the summary statistic is calculated

    Returns
    -------
    bs: xarray.Dataset
        Bier score, reliability, resolution
    y_prob_i: xarray.Dataarray
        Unique forecast values
    y_freq_i: xarray.Dataarray
        Conditional relative frequency
    n_i: xarray.Dataarray
        The number of each unique forecast value
    """
    if isinstance(dim, (list, np.ndarray)):
        y = y.stack(sample=dim)
        y_hat = y_hat.stack(sample=dim)
        dim = 'sample'

    # Tercile
    thresholds = y.quantile([1/3, 2/3], dim=dim) 

    # Forecast prob
    y_prob_low = (y_hat < thresholds[0]).mean(dim=ensemble_dim).rename('prob')
    y_prob_up = (y_hat > thresholds[1]).mean(dim=ensemble_dim).rename('prob')

    # Binary obs
    obs_low = y < thresholds[0]
    obs_up = y > thresholds[1]

    # Brier score
    bs_low, y_prob_i_low, y_freq_i_low, n_i_low = eval_bs(obs_low, y_prob_low)
    bs_up, y_prob_i_up, y_freq_i_up, n_i_up = eval_bs(obs_up, y_prob_up)

    # Concat
    labels = pd.Index(['Lower tercile', 'Upper tercile'], name='tercile')
    bs = xr.concat([bs_low, bs_up], labels)
    y_prob_i = xr.concat([y_prob_i_low, y_prob_i_up], labels)
    n_i = xr.concat([n_i_low, n_i_up], labels).fillna(0)
    y_freq_i = xr.concat([y_freq_i_low, y_freq_i_up], labels)

    if return_prob:
        return bs, y_prob_i, y_freq_i, n_i
    else:
        return bs

def eval_crps_decomp(ref, pred, dim='sample', ensemble_dim='analog'):
    """
    Decomposition of biased CRPS based on Hershbach (2020)

    Parameters
    ----------
    ref : xarray.DataArray (sample_dims,)
        The actual values
    pred : xarray.DataArray (ensemble, sample_dims,)
        The ensemble predicted values

    Returns
    -------
    ds : xarray.DataSet
        crps : (,)
            Sample-mean CRPS (biased)
        rel : (,)
            Reliability
        res : (,)
            Resolution
    """
    # Preprocess dimensions
    if isinstance(dim, (list, np.ndarray)):
        ref = ref.stack(sample=dim)
        pred = pred.stack(sample=dim)
    else:
        pred = pred.transpose(..., dim)
        if dim != 'sample':
            pred = pred.rename({dim: 'sample'})
            ref = ref.rename({dim: 'sample'})
    
    n_ens = pred[ensemble_dim].size
    n_sample = pred['sample'].size

    ref = ref.drop('sample').assign_coords(sample=np.arange(n_sample))
    pred = pred.drop('sample').assign_coords(sample=np.arange(n_sample))

    # Sort reference
    ref_sort = np.sort(ref)
    ref_sort = ref.copy(data=ref_sort).drop('sample').transpose('sample', ...)

    # Uncertainty term in CRPS
    weight_cum = xr.DataArray((np.arange(n_sample) / n_sample)[1:], dims=['sample'])
    unc = (weight_cum * (1 - weight_cum) * (ref_sort[1:] - ref_sort[:-1])
           ).sum(dim='sample', skipna=False)

    # Flatten
    if ref.ndim > 1:
        pred = pred.stack(all_dim=ref.dims)
        ref = ref.stack(all_dim=ref.dims)

    # Mask Nan
    mask = ref.notnull()
    ref = ref.where(mask, drop=True)
    pred = pred.where(mask, drop=True)

    # Sort forecast members
    pred_sort = np.sort(pred, axis=0)
    pred_sort = pred.copy(data=pred_sort)

    # Cumulative distribution
    p = xr.DataArray(np.linspace(0, 1, n_ens+1), dims=[ensemble_dim])

    # Difference between the i-th and (i+1)-th ensemble members
    binsize = pred_sort[1:] - pred_sort[:-1]

    # Initialize bin sizes used to calculate CRPS
    alpha = binsize.pad({ensemble_dim: (1, 1)}, constant_values=0)
    beta = binsize.pad({ensemble_dim: (1, 1)}, constant_values=0)

    alpha = alpha.where(
        (pred_sort < ref).pad({ensemble_dim: (0, 1)}, constant_values=False), 
        other=(ref - pred_sort).pad({ensemble_dim: (1, 0)}, constant_values=0))
    alpha = alpha.where(alpha > 0, other=0)

    beta = beta.where(
        (pred_sort > ref).pad({ensemble_dim: (1, 0)}, constant_values=False), 
        other=(pred_sort - ref).pad({ensemble_dim: (0, 1)}, constant_values=0))
    beta = beta.where(beta > 0, other=0)

    # Unmask
    _, alpha = xr.align(mask, alpha, join='left')
    _, beta = xr.align(mask, beta, join='left')

    # Sample-mean
    # print('sample-mean')
    alpham = alpha.unstack().mean(dim='sample')
    betam = beta.unstack().mean(dim='sample')
    g = alpham + betam
    # o = betam / g   # g[0]/g[-1] can be zero if there is no outlier
    # o = np.divide(betam, g, out=np.zeros_like(betam), where=g!=0)
    o = np.divide(betam.data, g.data, out=np.zeros_like(betam), where=g.data!=0)
    o = g.copy(data=o)

    # Sample-mean CRPS
    crps = (alpham * p ** 2 + betam * (1 - p) ** 2).sum(
        dim=ensemble_dim, skipna=False).rename('crps')
    crps.attrs['long_name'] = 'CRPS'

    # Reliability
    rel = (g * (o - p) ** 2).sum(dim=ensemble_dim, skipna=False).rename('rel')
    rel.attrs['long_name'] = 'Reliability'

    # Potential CRPS
    # pot = (g * o * (1 - o)).sum(dim=ensemble_dim, skipna=False).rename('pot')

    # Resolution
    res = (rel + unc - crps).rename('res')
    res.attrs['long_name'] = 'Resolution'

    return xr.merge([crps, rel, res])


def eval_crps_decomp_old(y, y_hat, dim='sample'):
    """
    Decomposition of biased CRPS based on Hershbach (2020)

    Only accepts 1d reference data. Slow.

    Parameters
    ----------
    y : array (sample_dims)
        The actual values
    y_hat : array (ensemble, sample_dims)
        The ensemble predicted values

    Returns
    -------
    crps : float
        Sample-mean CRPS (biased)
    rel : float
        Reliability
    res : float
        Resolution
    """
    if isinstance(dim, (list, np.ndarray)):
        y = y.stack(sample=dim)
        y_hat = y_hat.stack(sample=dim)
    else:
        y_hat = y_hat.transpose(..., dim)
    n_ens, n_sample = y_hat.shape

    # Sort forecast members
    y_hat_sort = np.sort(y_hat, axis=0)

    # Cumulative distribution
    p = np.linspace(0, 1, n_ens+1)

    # Difference between the i-th and (i+1)-th ensemble members
    binsize = y_hat_sort[1:] - y_hat_sort[:-1]

    # Initialize bin sizes used to calculate CRPS
    zero_array = np.zeros((1,) + y_hat.shape[1:])
    alpha = np.concatenate((zero_array, binsize, zero_array), axis=0)
    beta = np.concatenate((zero_array, binsize, zero_array), axis=0)

    for i in range(n_sample):
        # Position of observation in ensemble forecasts
        obs_idx = np.searchsorted(y_hat_sort[:, i], y[i])

        if obs_idx > 0:
            alpha[obs_idx, i] = y[i] - y_hat_sort[obs_idx-1, i]
        alpha[obs_idx+1:, i] = 0

        if obs_idx < n_ens:
            beta[obs_idx, i] = y_hat_sort[obs_idx, i] - y[i]
        beta[:obs_idx, i] = 0

    # Sample-mean
    alpham = alpha.mean(axis=1)
    betam = beta.mean(axis=1)
    g = alpham + betam
    # o = betam / g   # g[0]/g[-1] can be zero if there is no outlier
    o = np.divide(betam, g, out=np.zeros_like(betam), where=g!=0)

    # Sample-mean CRPS
    crps = (alpham * p ** 2 + betam * (1 - p) ** 2).sum()

    # Reliability
    rel = (g * (o - p) ** 2).sum()

    # Potential CRPS
    # pot = (g * o * (1 - o)).sum()

    # Uncertainty
    y_sort = np.sort(y)
    weight_cum = (np.arange(n_sample) / n_sample)[1:]
    unc = (weight_cum * (1 - weight_cum) * (y_sort[1:] - y_sort[:-1])).sum()

    # Resolution
    res = rel + unc - crps

    # Dataset
    ds = xr.Dataset({'crps': crps, 'rel': rel, 'res': res})
    ds['crps'].attrs['long_name'] = 'CRPS'
    ds['rel'].attrs['long_name'] = 'Reliability'
    ds['res'].attrs['long_name'] = 'Resolution'

    return ds

def eval_stats_lead(eval_func, ds, pred, dim=None, month=None):
    """
    Evaluate monthly forecast skills over lead times

    Parameters
    ----------
    eval_func : function
        Evaluation function that accepts (ref, pred, dim)
    ds : xarray (time, ...)
        Reference dataset
    pred : xarray, (lead, year, ...)
        The predicted value
    dim : str or list of str
        Dimensions along which the summary statistic is calculated
    month : int or None
        Month of data

    Returns
    -------
    stats : xarray (lead, ...)
    """
    
    lst = []
    for lead in pred.lead.data:
        # print(f'    {lead}-month lead')

        # Get reference
        ds_shift = shift_time(ds, lead)

        if month is None:
            ref = to_monthly(ds_shift)
        else:
            ref = extract_month_and_to_yearly(ds_shift, month)

        # Align dimension size
        ref, pred_lead = xr.align(ref, pred.sel(lead=lead))
        
        lst.append(eval_func(ref, pred_lead, dim))
    
    return xr.concat(lst, pred.lead).compute() 

# def eval_t_stats_month(ds, pred, dim, month=None):
#     """
#     Return collection of time-series skills
#     """

#     t_mse = eval_stats_lead(eval_mse, ds, pred, dim, month=month)
#     t_uac = eval_stats_lead(eval_uac, ds, pred, dim, month=month)
#     t_cac = eval_stats_lead(eval_r, ds, pred, dim, month=month)
#     t_rmsss = eval_stats_lead(eval_rmsss, ds, pred, dim, month=month)
#     t_msss = eval_stats_lead(eval_msss, ds, pred, dim, month=month)

#     # Combine
#     t_stats = xr.merge([
#         t_mse.rename('mse').assign_attrs(long_name='Mean square error'), 
#         t_uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation'),
#         t_cac.rename('cac').assign_attrs(long_name='Centered anomaly correlation'),
#         t_rmsss.rename('rmsss').assign_attrs(long_name='Root mean square skill score'),
#         t_msss.rename('msss').assign_attrs(long_name='Mean square skill score'),
#         ])
    
#     return t_stats

# def eval_xy_stats(ds, pred, dim, month=None):
#     """
#     Return collection of spatial skills
#     """

#     mse = eval_stats_lead(eval_mse, ds, pred, dim, month=month)
#     uac = eval_stats_lead(eval_uac, ds, pred, dim, month=month)

#     # Combine
#     stats = xr.merge([
#         mse.rename('mse').assign_attrs(long_name='Mean square error'), 
#         uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation'),
#         ])
    
#     return stats

# def eval_crps(pred, truth, sample_dim='analog'):
#     """
#     Parameters
#     ----------
#     pred : numpy.ndarray, (n_samples,) + truth.shape
#     truth : float or numpy.ndarray
    
#     Returns
#     -------
#     crps : truth.shape
#     """
#     n_samples = pred[sample_dim].size
#     absolute_error = np.abs(pred - truth).mean(dim=sample_dim)

#     if n_samples == 1:
#         return absolute_error

#     pred = pred.sortby(sample_dim)
#     weight = xr.DataArray(np.arange(n_samples), 
#                           coords={sample_dim: pred.coords[sample_dim]})
    
#     crps = (absolute_error + pred.mean(dim=sample_dim) 
#             - 2 * (weight * pred).sum(dim=sample_dim) 
#             / (n_samples * (n_samples-1)))

#     return crps.compute()