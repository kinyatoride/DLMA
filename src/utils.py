import numpy as np
import pandas as pd
import xarray as xr

class DotDict(dict):     
    """dot.notation access to dictionary attributes"""      
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val      
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def extract_month_and_to_yearly(ds, month):
    """
    Extract a specific month from an xarray time series
    and replace the time coordinate with the year
    """
    ds_month = ds.sel(time=(ds['time.month'] == month))
    return ds_month.assign_coords(time=ds_month['time.year']).rename({'time': 'year'})

def to_monthly(ds):
    """Reshape time dimension to (year, month)"""
    ds = ds.assign_coords(year=('time', ds.time.dt.year.data), 
                          month=('time', ds.time.dt.month.data))
    return ds.set_index(time=('year', 'month')).unstack('time')

def monthly_to_time(ds):
    """Stack (year, month) to time dimension (year-month-01)"""
    ds = ds.stack(time=('year', 'month'))
    dates = pd.DataFrame(list(ds.time.data), columns=['year', 'month'])
    dates['day'] = 1
    dates = pd.to_datetime(dates).rename('time')
    return ds.drop('time').assign_coords(time=('time', dates))

def shift_time_to_verification(ds):
    """
    Shift time from initial time to verification time

    Parameters
    ----------
    ds : xarray (time, lead)
        Predictions
    """
    t0 = pd.to_datetime(ds.time.data) 
    return xr.concat([ds.sel(lead=lead).assign_coords(
        {'time': t0 + pd.DateOffset(months=lead)})
        for lead in ds.lead.data], ds.lead)

def shift_time_to_initial(ds):
    """
    Shift time from verification time to initial time

    Parameters
    ----------
    ds : xarray (time, lead)
        Predictions
    """
    t0 = pd.to_datetime(ds.time.data) 
    return xr.concat([ds.sel(lead=lead).assign_coords(
        {'time': t0 - pd.DateOffset(months=lead)})
        for lead in ds.lead.data], ds.lead).dropna(dim='time', how='all')

def shift_time(ds, lead):
    """
    Shift time by -lead months
    """
    t0 = pd.to_datetime(ds.time.data) - pd.DateOffset(months=lead)
    return ds.assign_coords({'time': t0})

def stack_ens_year(ds, period, sample_dim='sample'):
    """
    For CESM2LENS dataset.
    Select period and stack the ens and year dimensions
    """
    return ds.sel(year=slice(*period)).stack(
        new_dim=['ens', 'year']).rename(
        {'new_dim': sample_dim}).transpose(sample_dim, ...)

def nino_indices(da):
    """
    Parameters
    ----------
    da : xarray.DataArray (lat, lon, )
        SST anomaly data
    
    Returns
    -------
    nino : xarray.Dataset
        Nino indices
    """
    wgt = np.cos(np.deg2rad(da.lat))
    
    nino3 = da.sel(lon=slice(210, 270), lat=slice(-5, 5)
                   ).weighted(wgt).mean(dim=['lat', 'lon']).assign_attrs(
            {'long_name': 'Niño 3', 'units': '°C'}
            ).rename('nino3')
    nino4 = da.sel(lon=slice(160, 210), lat=slice(-5, 5)
                   ).weighted(wgt).mean(dim=['lat', 'lon']).assign_attrs(
            {'long_name': 'Niño 4', 'units': '°C'}
            ).rename('nino4')
    nino34 = da.sel(lon=slice(190, 240), lat=slice(-5, 5)
                    ).weighted(wgt).mean(dim=['lat', 'lon']).assign_attrs(
            {'long_name': 'Niño 3.4', 'units': '°C'}
            ).rename('nino34')
    
    oni = nino34.rolling(time=3, center=True).mean().assign_attrs(
        {'long_name': 'ONI', 'units': '°C'}
        ).rename('oni')

    nino = xr.merge([nino3, nino4, nino34, oni]).compute()

    return nino

def est_p_fdr(p, alphag=0.05, multi=1):
    
    p_sort = np.sort(p[np.isfinite(p)], axis=None)

    n_test = len(p_sort)
    x = np.arange(1, n_test+1)
    fdr = x / n_test * alphag * multi # False discovery rate

    # Find first location where p is less than fdr
    N_fdr = np.searchsorted(p_sort - fdr, 0)
    p_fdr = p_sort[N_fdr] 

    return p_fdr

def seasonal_mean(ds, s_month):
    """
    Calculate seasonal mean from a starting month (s_month)
    """
    m2b = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    month_idx = [ds.month == month for month in range(1, 13)]
    
    season_idx = [
        month_idx[np.mod(s_month+i-1, 12)] 
        + month_idx[np.mod(s_month+i, 12)] 
        + month_idx[np.mod(s_month+i+1, 12)] 
        for i in range(0, 12, 3)
    ]
    
    labels = [
        m2b[np.mod(s_month+i-1, 12)] 
        + m2b[np.mod(s_month+i, 12)] 
        + m2b[np.mod(s_month+i+1, 12)]
        for i in range(0, 12, 3)
    ]
    
    season_mean = xr.concat([
        ds.sel(month=season_idx[0]).mean(dim='month'),
        ds.sel(month=season_idx[1]).mean(dim='month'),
        ds.sel(month=season_idx[2]).mean(dim='month'),
        ds.sel(month=season_idx[3]).mean(dim='month')],
        pd.Index(labels, name='season'))
    
    return season_mean

def convert_bytes(size):
    for units in ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0:
            return f'{size:3.1f} {units}'
        size /= 1024.0