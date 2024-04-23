import os
import sys
import json
import torch
import pandas as pd
import xarray as xr

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ml_only.loaddata import load_cesm2_by_period
from ml_only.UNet_dynamic import UNet
from ml_only.testing import test
from utils import DotDict
from analog import *
from eval_pred import *

argvs = sys.argv
if (len(argvs) != 2):
    print(f'Usage: python {argvs[0]} exp')
    quit()
exp = argvs[1]

# %%
# Parameters
out_dir = f'../output/{exp}'
data_dir = '../data/cesm2'
test_data = 'test'

with open(f'{out_dir}/hyperparameters.json', 'r') as f:
    hp = json.load(f)
    hp = DotDict(hp)

periods = {
    'library': hp.periods['train'],
    'target': hp.periods[test_data],
}

history = pd.read_csv(f'{out_dir}/history.csv', index_col=0)

# %%
# load data for U-Net
datasets, dataloaders, t1_wgt = load_cesm2_by_period(data_dir, **hp, shuffle=False)

# dimension
x, y = datasets['train'][0]
x_shape = tuple(x.shape)
y_shape = tuple(y.shape)
n_channels = x.shape[0]

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
t1_wgt = t1_wgt.to(device)

# Reference data
y = datasets[test_data].t1_ds

# Mask land
landmask = (y == 0).all(dim='sample')
y = y.where(~landmask)

def stats_all(y, y_pred, lead):
    # Time stats
    t_mse = eval_mse(y, y_pred, dim='sample')
    t_uac = eval_uac(y, y_pred, dim='sample')
    t_cac = eval_r(y, y_pred, dim='sample')
    t_rmsss = eval_rmsss(y, y_pred, dim='sample')
    t_msss = eval_msss(y, y_pred, dim='sample')

    # Over the target region
    xy_mse = eval_mse(y, y_pred, dim=['lat', 'lon'])
    xy_uac = eval_uac(y, y_pred, dim=['lat', 'lon'])

    # Combine
    t_stats = xr.merge([
        t_mse.rename('mse').assign_attrs(long_name='Mean square error'), 
        t_uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation'),
        t_cac.rename('cac').assign_attrs(long_name='Centered anomaly correlation'),
        t_rmsss.rename('rmsss').assign_attrs(long_name='Root mean square skill score'),
        t_msss.rename('msss').assign_attrs(long_name='Mean square skill score'),
        ])

    xy_stats = xr.merge([
        xy_mse.rename('mse').assign_attrs(long_name='Mean square error'), 
        xy_uac.rename('uac').assign_attrs(long_name='Uncentered anomaly correlation')
        ])

    # Assign lead time
    t_stats = t_stats.assign_coords({'lead': lead})
    xy_stats = xy_stats.assign_coords({'lead': lead})
    
    return t_stats, xy_stats.unstack()

# %%
lst = []
for epoch in range(hp.n_epochs):
    f = f'{out_dir}/model_epoch{epoch}.pt'
    if not os.path.exists(f):
        continue
    
    # Load model
    model = UNet(
        in_ch=n_channels, out_ch=1, 
        init_ch=hp.init_ch, depth=hp.depth,
        in_shape=x_shape, out_shape=y_shape,
        attention=hp.attention, is_res=hp.is_res,
        ).to(device)
    model.load_state_dict(torch.load(f))
    model.eval()

    # Test acc, get analog indices
    loss, y_pred = test(
        model, device, dataloaders[test_data], t1_wgt,
    )
    print(f'{exp} epoch {epoch:3d}: MSE = {loss:.3f}')

    # To xarray
    y_pred = xr.DataArray(y_pred.detach().cpu().numpy(), coords=y.coords)

    # Stats
    t_stats, xy_stats = stats_all(y, y_pred, hp.lead)

    # Save
    encoding = {key: {'dtype': 'float32'} for key in list(t_stats.keys())}
    t_stats.to_netcdf(f'{out_dir}/{hp.target_vname}_t_stats_{test_data}_epoch{epoch}.nc', encoding=encoding)

    encoding = {key: {'dtype': 'float32'} for key in list(xy_stats.keys())}
    xy_stats.to_netcdf(f'{out_dir}/{hp.target_vname}_xy_stats_{test_data}_epoch{epoch}.nc', encoding=encoding)