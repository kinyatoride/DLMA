import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

from utils import extract_month_and_to_yearly, shift_time, stack_ens_year

class AnalogDataset(Dataset):
    def __init__(self, 
                 t0_ds: xr.Dataset, 
                 t1_ds: xr.DataArray, 
                 t1_dist_ma: xr.DataArray):
        self.t0_ds = t0_ds.transpose('sample', 'lat', 'lon')
        self.t1_ds = t1_ds.transpose('sample', 'lat', 'lon')
        self.t1_dist_ma = t1_dist_ma
        self.vnames = list(t0_ds.keys())
        self.dims = self.t0_ds[self.vnames[0]].shape
        
    def __len__(self):
        return self.dims[0]
    
    def __getitem__(self, idx):
        
        # Inidital condition
        # Stack all variables
        x0 = torch.from_numpy(
            self.t0_ds.isel(sample=idx).to_array().data
        )

        # Target condition
        x1 = torch.from_numpy(
            self.t1_ds.isel(sample=idx).data
        )

        # Target distance
        t1_dist = torch.from_numpy(self.t1_dist_ma.isel(sample=idx).data)
        
        return x0, x1, t1_dist

def load_data(
    data_dir, vnames, lat_slice, 
    target_vname, target_grid, target_lat_slice, target_lon_slice,
    t1_dist_f, lead, month, sample_dim='year'
):
    """
    Parameters
    ----------
    data_dir : str
        Data directory
    vnames : list of str
        List of variable names
    lat_slice : tuple 
        Latitudinal bounds. e.g. (-60, 60)
    target_vname : str
        Target variable name
    target_grid : str
        Target grid resolution e.g. 2x2
    target_lat_slice : tuple
        Target latitudinal bounds
    target_lon_slice : tuple
        Target longitudinal bounds
    t1_dist_f : str
        File name of target distance matrix
    lead : int
        Forecast lead (months)
    month : int
        Initial month
    sample_dim : str or list of str
        Sample dimension
    
    Returns
    -------
    t0_ds_wgt_std : xarray.Dataset (sample, lat, lon)
        Initial states (weighted & scaled)
    t1_ds_wgt : xarray.DataArray (sample, lat, lon)
        Final states (weighted)
    t1_dist : xarray.DataArray (sample, sample)
        Final state distance matrix
    """
    # Read initial data
    f_lst = [f'{data_dir}/{vname}_anomaly_5x5.nc' for vname in vnames]
    t0_ds = xr.open_mfdataset(f_lst)
    t0_ds = t0_ds.sel(lat=slice(*lat_slice))
    
    # Read final data
    f = f'{data_dir}/{target_vname}_anomaly_{target_grid}.nc'
    t1_ds = xr.open_dataarray(f)
    t1_ds = t1_ds.sel(lat=slice(*target_lat_slice),
                      lon=slice(*target_lon_slice))

    # t1 distance matrix
    f = f'{data_dir}/{t1_dist_f}'
    t1_dist = xr.open_dataarray(f)

    # Shift the time of labels (t1 -> t0)
    t1_ds = shift_time(t1_ds, lead)
    # t0 = pd.to_datetime(t0_ds.time.data) - pd.DateOffset(months=lead)
    # t1_ds = t1_ds.assign_coords({'time': t0})

    # Select the month
    t0_ds_month = extract_month_and_to_yearly(t0_ds, month)
    t1_ds_month = extract_month_and_to_yearly(t1_ds, month)

    # Target month
    t1_month = (month + lead - 1) % 12 + 1
    t1_dist = t1_dist.sel(month=t1_month)
    
    # Shift the time of labels
    year_shift = (month + lead - 1) // 12 
    if year_shift > 0:
        year0 = t1_dist.year.data - year_shift
        lyear0 = t1_dist.lyear.data - year_shift
        t1_dist = t1_dist.assign_coords({'year': year0, 'lyear': lyear0})

    # Weight by sqrt(cos(lat)) ~ sqrt(grid area)
    t0_wgt = np.sqrt(np.cos(np.deg2rad(t0_ds.lat))).astype('float32')
    t1_wgt = np.sqrt(np.cos(np.deg2rad(t1_ds.lat))).astype('float32')

    if isinstance(sample_dim, str):
        t0_wgt_mean = t0_wgt.where(t0_ds_month.isel({sample_dim: 0}).notnull()).mean()
        t1_wgt_mean = t1_wgt.where(t1_ds_month.isel({sample_dim: 0}).notnull()).mean()
    else:
        t0_wgt_mean = t0_wgt.where(t0_ds_month.isel({dim: 0 for dim in sample_dim}).notnull()).mean()
        t1_wgt_mean = t1_wgt.where(t1_ds_month.isel({dim: 0 for dim in sample_dim}).notnull()).mean()

    t0_ds_wgt = (t0_ds_month * t0_wgt / t0_wgt_mean).compute()
    t1_ds_wgt = (t1_ds_month * t1_wgt / t1_wgt_mean).compute()
    
    # Scale by domain-averaged monthly std
    # Suppress runtime warning for empty array
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        std = t0_ds_wgt.var(dim=sample_dim).mean(dim=['lat', 'lon']) ** 0.5
    t0_ds_wgt_std = (t0_ds_wgt / std).compute()
    
    return t0_ds_wgt_std, t1_ds_wgt, t1_dist

def load_cesm2_by_period(
    data_dir, vnames, lat_slice, 
    target_vname, target_grid, target_lat_slice, target_lon_slice,
    t1_dist_f, lead, month, periods, 
    batch_size, shuffle=True, **kwargs
):
    
    t0_ds, t1_ds, t1_dist = load_data(
        data_dir, vnames, lat_slice, 
        target_vname, target_grid, target_lat_slice, target_lon_slice,
        t1_dist_f, lead, month, sample_dim=['ens', 'year'])
    
    datasets = {}
    dataloaders = {}
    for key, period in periods.items():
        
        # Split data, stack the ens and year dimensions to a sample dimension
        t0_ds_sel = stack_ens_year(t0_ds, period)
        t1_ds_sel = stack_ens_year(t1_ds, period)
        t1_dist_sel = t1_dist.sel(
            year=slice(*period), lyear=slice(*periods['train'])
        ).stack(sample=('ens', 'year'), lsample=('lens', 'lyear'))
        
        # Replace NaN with 0
        t0_mask = torch.from_numpy(t0_ds_sel.to_array().isel(sample=0).isnull().data)
        t0_ds_sel = t0_ds_sel.fillna(0)
        
        # Create the library in tensor
        if key == 'train':
            t0_library = torch.from_numpy(t0_ds_sel.to_array().transpose('sample', ...).data)
            t1_library = torch.from_numpy(t1_ds_sel.transpose('sample', ...).data)
    
        datasets[key] = AnalogDataset(t0_ds_sel, t1_ds_sel, t1_dist_sel)
        dataloaders[key] = DataLoader(datasets[key],
                                      batch_size=batch_size,
                                      shuffle=shuffle)

    return datasets, dataloaders, t0_library, t0_mask, t1_library

def load_real(
    data_dir, vnames, lat_slice, 
    target_vname, target_grid, target_lat_slice, target_lon_slice,
    t1_dist_f, lead, month, periods, 
    batch_size, shuffle=False, **kwargs
):
    
    t0_ds, t1_ds, t1_dist = load_data(
        data_dir, vnames, lat_slice, 
        target_vname, target_grid, target_lat_slice, target_lon_slice,
        t1_dist_f, lead, month)
        
    # Select period, rename time to sample
    t0_ds_sel = t0_ds.sel(year=slice(*periods['test'])).rename({'year': 'sample'})
    t1_ds_sel = t1_ds.sel(year=slice(*periods['test'])).rename({'year': 'sample'})
    t1_dist_sel = t1_dist.sel(
        year=slice(*periods['test']), 
        lyear=slice(*periods['train'])
    ).stack(lsample=('lens', 'lyear')).rename({'year': 'sample'})
    
    # Replace NaN with 0
    t0_mask = torch.from_numpy(t0_ds_sel.to_array().isel(sample=0).isnull().data)
    t0_ds_sel = t0_ds_sel.fillna(0)

    dataset = AnalogDataset(t0_ds_sel, t1_ds_sel, t1_dist_sel)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    return dataset, dataloader, t0_mask

def load_cesm2_by_ens(
    data_dir, vnames, lat_slice, 
    t1_dist_f, lead, month, ens_dict, 
    batch_size, shuffle=True
):
    
    t0_ds, t1_ds, t1_dist = load_data(
        data_dir, vnames, lat_slice, t1_dist_f, 
        lead, month, sample_dim=['ens', 'year'])

    # Select syear to eyear-2
    syear = t0_ds.year.data[0]
    eyear = t0_ds.year.data[-3]
    t0_ds = t0_ds.sel(year=slice(syear, eyear))
    t1_ds = t1_ds.sel(year=slice(syear, eyear))
    t1_dist = t1_dist.sel(year=slice(syear, eyear), lyear=slice(syear, eyear))
    
    datasets = {}
    dataloaders = {}
    for key, ens in ens_dict.items():
        
        # Split data, stack the ens and year dimensions to a sample dimension
        t0_ds_sel = t0_ds.sel(ens=ens).stack(sample=('ens', 'year')).transpose('sample', ...)
        t1_ds_sel = t1_ds.sel(ens=ens).stack(sample=('ens', 'year')).transpose('sample', ...)
        t1_dist_sel = t1_dist.sel(
            ens=ens, lens=ens_dict['train']
        ).stack(sample=('ens', 'year'), lsample=('lens', 'lyear'))
        
        # Replace NaN with 0
        t0_mask = torch.from_numpy(t0_ds_sel.to_array().isel(sample=0).isnull().data)
        t0_ds_sel = t0_ds_sel.fillna(0)
        
        # Create the library in tensor
        if key == 'train':
            t0_library = torch.from_numpy(t0_ds_sel.to_array().transpose('sample', ...).data)
            t1_library = torch.from_numpy(t1_ds_sel.transpose('sample', ...).data)
    
        datasets[key] = AnalogDataset(t0_ds_sel, t1_ds_sel, t1_dist_sel)
        dataloaders[key] = DataLoader(datasets[key],
                                      batch_size=batch_size,
                                      shuffle=shuffle)

    return datasets, dataloaders, t0_library, t0_mask, t1_library
