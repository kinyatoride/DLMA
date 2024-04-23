import os
import json
import numpy as np
import pandas as pd
import xarray as xr

def load_stats(stats, exps, exp_names, epochs, vname, test_data, month):
    lst = []

    for exp, epoch in zip(exps, epochs):
        out_dir = f'../output/{exp}'
        if 'MA' in exp:
            if stats == 'xy_stats':
                f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
            else:
                f = f'{out_dir}/{vname}_{stats}_month_{test_data}.nc'
            ds = xr.open_dataset(f).sel(month=month)
        else:
            if epoch is None:
                if not os.path.exists(f'{out_dir}/history.csv'):
                    f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
                else:
                    history = pd.read_csv(f'{out_dir}/history.csv', index_col=0)
                    epoch = history['val_mse'].argmin()
                    f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
            else:
                f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
            ds = xr.open_dataset(f)

        lst.append(ds)
    ds = xr.concat(lst, pd.Index(exp_names, name='exp')).transpose(..., 'exp')    

    return ds


def load_stats_ens(stats, exps, exp_names, common_epochs, vname, test_data, month, n_f_ens):
    lst = []

    for exp, common_epoch in zip(exps, common_epochs):
        if 'MA' in exp:
            out_dir = f'../output/{exp}'
            f = f'{out_dir}/{vname}_{stats}_month_{test_data}.nc'
            ds_base = xr.open_dataset(f).sel(month=month)
        elif 'gens' in exp:  # Grand ensemble
            out_dir = f'../output/{exp}'
            if common_epoch is None:
                f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
            else:
                f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{common_epoch}.nc'
            lst.append(xr.open_dataset(f))
        else:
            lst_ens = []
            for i in range(n_f_ens):
                out_dir = f'../output/{exp}_{i}'
                if common_epoch is None:
                    if not os.path.exists(f'{out_dir}/history.csv'):
                        f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
                    else:
                        history = pd.read_csv(f'{out_dir}/history.csv', index_col=0)
                        epoch = history['val_mse'].argmin()
                        f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
                else:
                    f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{common_epoch}.nc'
                
                if not os.path.exists(f):
                    continue

                lst_ens.append(xr.open_dataset(f))

            lst.append(xr.concat(lst_ens, pd.Index(np.arange(len(lst_ens)), name='f_ens')))
    
    if 'MA' in exps:
        ds = xr.concat(lst, pd.Index(exp_names[1:], name='exp')).transpose(..., 'exp')    
        return ds_base, ds
    else:
        ds = xr.concat(lst, pd.Index(exp_names, name='exp')).transpose(..., 'exp')    
        return ds

def load_ml_stats_ens(stats, exps, common_epochs, vname, test_data, n_f_ens):
    lst = []
    for exp, common_epoch in zip(exps, common_epochs):

        lst_ens = []
        for i in range(n_f_ens):
            out_dir = f'../output/{exp}_{i}'
            if not os.path.exists(f'{out_dir}/history.csv'):
                continue
            if common_epoch is None:
                history = pd.read_csv(f'{out_dir}/history.csv', index_col=0)
                epoch = history['val_loss'].argmin()
            else:
                epoch = common_epoch

            f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
            # print(f)

            if not os.path.exists(f):
                continue

            lst_ens.append(xr.open_dataset(f))

        lst.append(xr.concat(lst_ens, pd.Index(np.arange(len(lst_ens)), name='f_ens')))

    ds = xr.concat(lst, dim='lead').transpose('f_ens', ...)
    return ds

def load_ml_old_stats_ens(exps, common_epochs, test_data, n_f_ens):
    lst = []
    for exp, common_epoch in zip(exps, common_epochs):

        lst_ens = []
        for i in range(n_f_ens):
            out_dir = f'../output/{exp}_{i}'
            if not os.path.exists(f'{out_dir}/history_{test_data}.csv'):
                continue
            if common_epoch is None:
                history = pd.read_csv(f'{out_dir}/history_{test_data}.csv', index_col=0)
                epoch = history['val_loss'].argmin()
            else:
                epoch = common_epoch

            lst_ens.append(history.loc[epoch, 'test_loss'])

        with open(f'{out_dir}/hyperparameters.json', 'r') as f:
            hp = json.load(f)
        da = xr.DataArray(lst_ens, coords={'f_ens': np.arange(len(lst_ens)), 'lead': hp['lead']},
                          dims=['f_ens']) 
        lst.append(da)
    da = xr.concat(lst, dim='lead').transpose('f_ens', 'lead')
    return da

def load_stats_ens_month(stats, exps, exp_names, exp_months, common_epoch, vname, test_data, n_f_ens):

    lst = []
    for exp in exps:
        if 'MA' in exp:
            # MA
            out_dir = f'../output/{exp}'
            f = f'{out_dir}/{vname}_{stats}_month_{test_data}.nc'
            ds_base = xr.open_dataset(f).transpose(..., 'lead') 
        elif 'gens' in exp:
            # MA + ML (grand ensembles)
            lst_month = []
            for exp_month in exp_months:
                out_dir = f'../output/{exp_month}_gens0-9'
                if common_epoch is None:
                    f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
                else:
                    f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
                    epoch = common_epoch        
                lst_month.append(xr.open_dataset(f))
            lst.append(xr.concat(lst_month, pd.Index(np.arange(1, 13), name='month')))
        else:
            # MA + ML
            lst_month = []
            for exp_month in exp_months:
                lst_ens = []
                for i in range(n_f_ens):
                    out_dir = f'../output/{exp_month}_{i}'
                    if common_epoch is None:
                        history = pd.read_csv(f'{out_dir}/history.csv', index_col=0)
                        epoch = history['val_mse'].argmin()
                    else:
                        epoch = common_epoch
                    
                    f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
                    if not os.path.exists(f):
                        print(f)
                        continue
                    lst_ens.append(xr.open_dataset(f))
                lst_month.append(xr.concat(lst_ens, pd.Index(np.arange(len(lst_ens)), name='f_ens')))
            lst.append(xr.concat(lst_month, pd.Index(np.arange(1, 13), name='month')))

    if 'MA' in exps:
        ds = xr.concat(lst, pd.Index(exp_names[1:], name='exp')).transpose(..., 'exp')    
        return ds_base, ds
    else:
        ds = xr.concat(lst, pd.Index(exp_names, name='exp')).transpose(..., 'exp')    
        return ds

def load_stats_month(stats, exps, exp_names, vname, test_data):

    lst = []

    for exp in exps:
        out_dir = f'../output/{exp}'
        if 'MA' in exp:
            f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
            ds = xr.open_dataset(f)
        lst.append(ds)
    ds = xr.concat(lst, pd.Index(exp_names, name='exp')).transpose(..., 'exp')    

    return ds

# def load_stats_month(stats, exps, exp_months, common_epoch, vname, test_data):
    # # MA
    # out_dir = f'../output/MA'
    # f = f'{out_dir}/{vname}_{stats}_month_{test_data}.nc'
    # ds_base = xr.open_dataset(f) 

    # # MA + ML (grand ensembles)
    # lst_month = []
    # for exp_month in exp_months:
    #     out_dir = f'../output/{exp_month}_gens0-9'
    #     if common_epoch is None:
    #         f = f'{out_dir}/{vname}_{stats}_{test_data}.nc'
    #     else:
    #         f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{common_epoch}.nc'
    #     lst_month.append(xr.open_dataset(f))
    # ds = xr.concat(lst_month, pd.Index(np.arange(1, 13), name='month'))      

    # return ds_base, ds

def load_val_diff(stats, exps, med_f_enss, out_dir, vname, test_data, epoch):
    lst = []
    for exp, med_f_ens in zip(exps, med_f_enss):
        if 'MA' in exp:
            f = f'{out_dir}/{vname}_{stats}_{test_data}_epoch{epoch}.nc'
        else:
            f = f'{out_dir}/{vname}_{stats}_{test_data}_{exp}_{med_f_ens}.nc'

        lst.append(xr.open_dataset(f))
        
    ds = xr.concat(lst, pd.Index(exps, name='exp')).transpose(..., 'exp')   

    return ds