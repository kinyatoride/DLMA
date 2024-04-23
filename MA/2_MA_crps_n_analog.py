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
xr.set_options(keep_attrs=True)

# %%
# Parameters
test_data = 'test'
out_dir = '../output/MA'
n_analogs = np.arange(5, 101)
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
af = get_af(da, param.periods.library, ma_idx, n_analogs[-1], leads)
#lst = []
#for lead in leads:
#   print(f'{lead}-month lead')
#   af = get_af(da, param.periods.library, ma_idx, n_analogs[-1], [lead])
#   lst.append(af)
#af = xr.concat(lst, dim='lead')


print('CRPS')
lst = []
for n_analog in n_analogs:
    print(f'Analog member size: {n_analog}')
    # crps = eval_stats_lead(
    #     eval_crps_decomp, da, 
    #     af.sel(analog=slice(0, n_analog)),
    #     dim=['ens', 'year'])
    # lst.append(crps)

    lst_month = []
    for month in ma_idx.month.data:
        print(f'  Initial month: {month}')
        crps = eval_stats_lead(
            eval_crps_decomp, da, 
            af.sel(analog=slice(0, n_analog), month=month),
            dim=['ens', 'year'], month=month)
        lst_month.append(crps)
    lst.append(xr.concat(lst_month, dim='month'))
crps = xr.concat(lst, pd.Index(n_analogs, name='n_analog'))

# Save
encoding = {key: {'dtype': 'float32'} for key in list(crps.keys())}
crps.to_netcdf(f'{out_dir}/{vname}_t_crps_month_{test_data}_n_analog.nc', encoding=encoding)


# # Nino 3.4 skills
#if vname == 'sst':
#    print('Nino 3.4')
#    nino = nino_indices(da)['nino34']
#    nino_af = get_af(nino, param.periods.library, ma_idx, n_analogs[-1], leads)
# 
#    lst = []
#    for n_analog in n_analogs:
#        print(f'Analog member size: {n_analog}')
#        crps = eval_stats_lead(
#            eval_crps_decomp, nino, 
#            nino_af.sel(analog=slice(0, n_analog)),
#            dim=['ens', 'year'])
#        lst.append(crps)
#    nino_crps = xr.concat(lst, pd.Index(n_analogs, name='n_analog'))
# 
#    # Save
#    encoding = {key: {'dtype': 'float32'} for key in list(nino_crps.keys())}
#    nino_crps.to_netcdf(f'{out_dir}/nino34_t_crps_month_{test_data}_n_analog.nc', encoding=encoding)
