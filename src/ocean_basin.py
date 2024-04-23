"""
Modules related to ocean basins
"""

import numpy as np
import pandas as pd
import xarray as xr

def basin_mask(landmask):
    """
    Return ocean basins (Indian, Pacific, and Atlantic)
    Limited to 60°S-60°N
    
    Parameters
    ----------
    landmask : xarray.DataArray
    
    Returns
    -------
    basins : xarray.DataArray
        1: Indian, 2: Pacific, 3: Atlantic
    """
    # 2D landmask field
    landmask = landmask.sel(lat=slice(-60, 60))
    lat = landmask.lat
    lon = landmask.lon

    # Indian
    basins = xr.where(
        (((lat >= 10) & (lon >= 30) & (lon < 99)) |
        ((lat < 10) & (lat >= -5) & (lon >= 30) & (lat < -15/6 * (lon-99) + 10)) |
        ((lat < -5) & (lat > -10) & (lon >= 30) & (lat < -5/15 * (lon-105) - 5)) |
        ((lat <= -10) & (lon >= 20) & (lon < 135))
        ), 
        1, 
        landmask)

    # Pacific
    basins = xr.where(
        (((lat >= 10) & (lon >= 99)) |
        ((lat < 10) & (lat >= -5) & (lon >= 30) & (lat >= -15/6 * (lon-99) + 10)) |
        ((lat < -5) & (lat > -10) & (lon >= 30) & (lat >= -5/15 * (lon-105) - 5)) |
        ((lat <= -10) & (lon >= 135))
        ), 
        2, 
        basins)

    # Atlantic
    basins = xr.where(
        (((lat >= 30) & (lon <= 45)) |
        ((lat >= 20) & (lon >= 260)) |
        ((lat < 20) & (lat >= 12) & (lat >= -8/15 * (lon-260) + 20)) |
        ((lat < 12) & (lat >= 9) & (lat >= -3/3 * (lon-275) + 12)) |
        ((lat < 9) & (lat >= 8) & (lat >= -1/17 * (lon-278) + 9)) |
        ((lat < 8) & (lon > 290)) |
        ((lon >=0) & (lon < 20))
        ), 
        3, 
        basins)

    basins = basins.where(~np.isnan(landmask)).rename('basin')
    
    return basins

def basin_sum(ds, basins):
    """
    Take sum over each basin
    
    Parameters
    ----------
    ds : xarray

    basins : xarray
        1: Indian, 2: Pacific, 3: Atlantic  
    """
    wgt = np.cos(np.deg2rad(ds.lat))
    
    lst = []
    for i in range(1, 4):
        lst.append(ds.where(basins == i).weighted(wgt).sum(dim=['lat', 'lon']))
        
    return xr.concat(lst, pd.Index(['Indian', 'Pacific', 'Atlantic'], name='basin'))

def basin_usum(ds, basins):
    """
    Take unweighted sum over each basin
    
    Parameters
    ----------
    ds : xarray

    basins : xarray
        1: Indian, 2: Pacific, 3: Atlantic  
    """    
    lst = []
    for i in range(1, 4):
        lst.append(ds.where(basins == i).sum(dim=['lat', 'lon']))
        
    return xr.concat(lst, pd.Index(['Indian', 'Pacific', 'Atlantic'], name='basin'))
    