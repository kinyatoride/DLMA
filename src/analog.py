"""
Modules for analog forecasting
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from utils import extract_month_and_to_yearly, shift_time, stack_ens_year

def get_af_month(ds, month, period, idx, n_analog, leads):
    """
    Return forward analog forecasts for a specific month
    
    Parameters
    ----------
    ds : xarray dataset or dataarray
        Library to take trajectories
    month : int
        Target month
    period: tuple
        Library period eg. (1900, 1950)
    idx : xarray dataset (analog,)
        Sorted analog indices 
    n_analog : int
        Number of analogs 
    leads : list of int
        lead months
    
    Returns
    -------
    af : xarray (analog,)
        Analog forecasts
    """

    lst_af = []

    for lead in leads:    
        # Get library
        ds_shift = shift_time(ds, lead)
        ds_month = extract_month_and_to_yearly(ds_shift, month)
        library = stack_ens_year(ds_month, period, sample_dim='lsample')
        library = library.drop('lsample')
        
        # analog forecasts
        af = library.isel(lsample=idx.isel(analog=slice(0, n_analog))
                          ).transpose('analog', ...)
        
        # Append
        lst_af.append(af)
    
    # concat
    af = xr.concat(lst_af, pd.Index(leads, name='lead'))              

    return af

def get_af(ds, period, idx, n_analog, leads):
    """
    Return forward analog forecasts for all months
    
    Parameters
    ----------
    ds : xarray dataset or dataarray
        Library to take trajectories
    period: tuple
        Library period eg. (1900, 1950)
    idx : xarray dataset (month, analog,)
        Sorted analog indices 
    n_analog : int
        Number of analogs
    leads : list of int
        lead months
    
    Returns
    -------
    af : xarray (month, analog,)
        Analog forecasts
    """

    lst_af = []
    for month in idx.month.data:
        af = get_af_month(ds, month, period, idx.sel(month=month), n_analog, leads)
        lst_af.append(af)
    
    # concat
    af = xr.concat(lst_af, idx.month)              
    
    return af

# def calc_mse_lead(pred, ds, period=None, month=None, dim=['lat', 'lon']):
#     """
#     Calculate mean-squared-error
#     """
    
#     # grid weight
#     if 'lat' in dim:
#         wgt = np.cos(np.deg2rad(ds.lat))  
    
#     lst_mse = []
#     for lead in pred.lead.data:
#         # Get reference
#         ds_shift = shift_time(ds, lead)
#         ref = to_monthly(ds_shift)

#         if 'ens' in ds.dims:
#             # For CESM2
#             ref = stack_ens_year(ref, period)
        
#         if month is not None:
#             ref = ref.sel(month=month)
        
#         pred_lead = pred.sel(lead=lead)  
        
#         # mse
#         if 'lat' in dim:
#             mse = ((pred_lead - ref) ** 2).weighted(wgt).mean(dim=dim)    
#         else:
#             mse = ((pred_lead - ref) ** 2).mean(dim=dim)    
        
#         # Append
#         lst_mse.append(mse)
    
#     # concat
#     mse = xr.concat(lst_mse, pred.lead).compute() 

#     return mse

# def calc_acu_lead(pred, ds, period=None, month=None, dim='sample'):
#     """
#     Calculate uncentered anomaly correlation
#     """
 
#     # grid weight
#     if 'lat' in dim:
#         wgt = np.cos(np.deg2rad(ds.lat))  
               
#     lst_acu = []
#     for lead in pred.lead.data:    
#         # Get reference
#         ds_shift = shift_time(ds, lead)
#         ref = to_monthly(ds_shift)

#         if 'ens' in ds.dims:
#             ref = stack_ens_year(ref, period)
            
#         if month is not None:
#             ref = ref.sel(month=month)
            
#         pred_lead = pred.sel(lead=lead)

#         # # Align dimension size
#         # ref, pred_lead = xr.align(ref, pred_lead)

#         # acu
#         if 'lat' in dim:
#             acu = ((ref * pred_lead).weighted(wgt).sum(dim=dim)
#                    / ((ref**2).weighted(wgt).sum(dim=dim) 
#                       * (pred_lead**2).weighted(wgt).sum(dim=dim)) ** 0.5)
#         else:
#             acu = ((ref * pred_lead).sum(dim=dim) 
#                    / ((ref**2).sum(dim=dim) 
#                       * (pred_lead**2).sum(dim=dim)) ** 0.5)
        
#         # Append
#         lst_acu.append(acu)
    
#     # concat
#     # Suppress runtime warning for empty array
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore', category=RuntimeWarning)
#         acu = xr.concat(lst_acu, pred.lead).compute()

#     return acu