# %% [markdown]
# # Calculate forcast skills of MA

# %%
import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from analog import *
from eval_pred import *
from utils import DotDict, nino_indices

# %%
# Parameters
test_data = 'test'
out_dir = '../output/MA'
n_analog = 30
leads = np.arange(19)
vname = 'sst'

# Target region to evaluate spatial skills
lat_slice = (-10, 10)
lon_slice = (120, 290)

# Load additional parameters
with open(f'{out_dir}/param_{test_data}.json', 'r') as f:
    param = json.load(f)
    param = DotDict(param)

# %%
# load data
if vname == 'pr':
    grid = '2.5x2.5'
else:
    grid = '2x2'
    
f = f'{param.data_dir}/{vname}_anomaly_{grid}.nc'
da = xr.open_dataarray(f)
da = da.sel(lat=slice(*lat_slice), lon=slice(*lon_slice))

# load MA indices
ma_idx = xr.open_dataarray(f'{out_dir}/ma_index_{test_data}.nc')  

# %%
print('Get analog forecasts')
af = get_af(da, param.periods.library, ma_idx, n_analog, leads)  # Memory issue
# lst = []
# for lead in leads:
#     af = get_af(da, param.periods.library, ma_idx, n_analog, [lead])
#     lst.append(af.mean(dim='analog'))
# afm = xr.concat(lst, dim='lead')

# %%
# # All-month stats
# print('All-month stats')
# t_mse = eval_stats_lead(eval_mse, da, afm, dim=['ens', 'year', 'month'])
# t_uac = eval_stats_lead(eval_uac, da, afm, dim=['ens', 'year', 'month'])

# # Monthly stats
# print('Monthly stats')
# t_mse_month = eval_stats_lead(eval_mse, da, afm, dim=['ens', 'year'])
# t_uac_month = eval_stats_lead(eval_uac, da, afm, dim=['ens', 'year'])
# t_cac_month = eval_stats_lead(eval_r, da, afm, dim=['ens', 'year'])
# t_rmsss_month = eval_stats_lead(eval_rmsss, da, afm, dim=['ens', 'year'])
# t_msss_month = eval_stats_lead(eval_msss, da, afm, dim=['ens', 'year'])

# # Over the target region
# print('Spatial stats')
# xy_mse = eval_stats_lead(
#     eval_mse, da.sel(lat=slice(*lat_slice), lon=slice(*lon_slice)), afm, dim=['lat', 'lon'])
# xy_uac = eval_stats_lead(
#     eval_uac, da.sel(lat=slice(*lat_slice), lon=slice(*lon_slice)), afm, dim=['lat', 'lon'])

# # Probablistic stats
# print('CRPS')
# t_crps_month = eval_stats_lead(eval_crps_decomp, da, af, dim=['ens', 'year'])

# Ensemble spread (time-mean)
print('Ensemble spread')
t_std_month = af.var(dim='analog').mean(dim=['ens', 'year']) ** 0.5
t_std_month = t_std_month.rename('std').assign_attrs(long_name='Ensemble spread') 

# %%
# # Combine
# t_stats = xr.merge([
#     t_mse.rename('mse').assign_attrs(long_name='Mean square error'), 
#     t_uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation')
#     ])

# t_stats_month = xr.merge([
#     t_mse_month.rename('mse').assign_attrs(long_name='Mean square error'), 
#     t_uac_month.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation'),
#     t_cac_month.rename('cac').assign_attrs(long_name='Centered anomaly correlation'),
#     t_rmsss_month.rename('rmsss').assign_attrs(long_name='Root mean square skill score'),
#     t_msss_month.rename('msss').assign_attrs(long_name='Mean square skill score'),
#     ])

# xy_stats = xr.merge([
#     xy_mse.rename('mse').assign_attrs(long_name='Mean square error'), 
#     xy_uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation')
#     ])

# %%
# # Save
print('save')
# encoding = {key: {'dtype': 'float32'} for key in list(t_stats.keys())}
# t_stats.to_netcdf(f'{out_dir}/{vname}_t_stats_{test_data}.nc', encoding=encoding)

# encoding = {key: {'dtype': 'float32'} for key in list(t_stats_month.keys())}
# t_stats_month.to_netcdf(f'{out_dir}/{vname}_t_stats_month_{test_data}.nc', encoding=encoding)

# encoding = {key: {'dtype': 'float32'} for key in list(xy_stats.keys())}
# xy_stats.to_netcdf(f'{out_dir}/{vname}_xy_stats_{test_data}.nc', encoding=encoding)

# encoding = {key: {'dtype': 'float32'} for key in list(t_crps_month.keys())}
# t_crps_month.to_netcdf(f'{out_dir}/{vname}_t_crps_month_{test_data}.nc', encoding=encoding)

encoding = {'std': {'dtype': 'float32'}}
t_std_month.to_netcdf(f'{out_dir}/{vname}_t_std_month_{test_data}.nc', encoding=encoding)

# %% [markdown]
# # Nino indices skills
# If vname == 'sst'

# %%
if vname == 'sst':
    print('nino')
    nino = nino_indices(da)
    nino_af = get_af(nino, param.periods.library, ma_idx, n_analog, leads)
    nino_afm = nino_af.mean(dim='analog')

    # nino_t_uac = eval_stats_lead(eval_uac, nino, nino_afm, dim=['ens', 'year', 'month'])
    # nino_t_uac_month = eval_stats_lead(eval_uac, nino, nino_afm, dim=['ens', 'year'])
    # nino_t_mse_month = eval_stats_lead(eval_mse, nino, nino_afm, dim=['ens', 'year'])
    # nino_t_rmsss_month = eval_stats_lead(eval_rmsss, nino, nino_afm, dim=['ens', 'year'])
    # nino_t_crps_month = eval_stats_lead(eval_crps_decomp, nino['nino34'], nino_af['nino34'], dim=['ens', 'year'])
    
    nino_t_std_month = nino_af.var(dim='analog').mean(dim=['ens', 'year']) ** 0.5

    # Save
    encoding = {key: {'dtype': 'float32'} for key in list(nino.keys())}
    # nino_t_uac.to_netcdf(f'{out_dir}/nino_t_uac_{test_data}.nc', encoding=encoding)
    # nino_t_uac_month.to_netcdf(f'{out_dir}/nino_t_uac_month_{test_data}.nc', encoding=encoding)
    # nino_t_mse_month.to_netcdf(f'{out_dir}/nino_t_mse_month_{test_data}.nc', encoding=encoding)
    # nino_t_rmsss_month.to_netcdf(f'{out_dir}/nino_t_rmsss_month_{test_data}.nc', encoding=encoding)
    nino_t_std_month.to_netcdf(f'{out_dir}/nino_t_std_month_{test_data}.nc', encoding=encoding)

    # encoding = {key: {'dtype': 'float32'} for key in list(nino_t_crps_month.keys())}
    # nino_t_crps_month.to_netcdf(f'{out_dir}/nino34_t_crps_month_{test_data}.nc', encoding=encoding)
