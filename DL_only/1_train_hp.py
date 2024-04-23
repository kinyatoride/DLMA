"""
Hyperparameter search
"""
import os
import sys
import json
import torch
import wandb 
import argparse

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ml_loaddata import load_cesm2_by_period
from UNet_dynamic import UNet
from ml_trainer import Trainer, EarlyStopper

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=1, help='Initial month')
parser.add_argument('--lead', type=int, default=12, help='Targed lead time (months)')
parser.add_argument('--depth', type=int, default=4, help='Number of layers in UNet')
parser.add_argument('--init-ch', type=int, default=256, help='Initial channel size in UNet')
parser.add_argument('--n-epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--count', type=int, default=30, help='Number of hyperparameter search')
args = parser.parse_args()

# Sweep config
project = 'ML-ENSO'
sweep_config = {
    'name': f'ml_month{args.month:02d}_lead{args.lead}',    # Sweep-name
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize',
    },
}

# Hyperparameters
hp = {
    'vnames': {
        'value': ['sst', 'ssh', 'taux']
        },
    'lat_slice': {
        'value': (-50, 50)
        },
    'target_vname': {'value': 'sst'},
    'target_grid': {'value': '2x2'},
    'target_lat_slice': {'value': (-10, 10)},
    'target_lon_slice': {'value': (120, 290)},
    'lead': {'value': args.lead},
    'month': {'value': args.month},
    'periods': {'value': {'train': (1865, 1958),
                          'val': (1959, 1985),
                          'test': (1986, 1998),
                          }},
    'batch_size': {'value': 16},
    'learning_rate': {
        'min': 1.0e-6,
        'max': 1.0e-4,
        'distribution': 'log_uniform_values',
    },
    'model': {'value': 'UNet'},
    'attention': {'value': False},
    'is_res': {'value': False},
    'depth' : {'value': args.depth},
    'init_ch': {'value': args.init_ch},
    'n_epochs': {'value': args.n_epochs},
}
data_dir = '../data/cesm2'

sweep_config['parameters'] = hp
sweep_id = wandb.sweep(sweep_config, project=project)

def train_sweep(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        # Change run name
        run_number = run.name.split('-')[-1]
        run.name = f'{sweep_config["name"]}_sweep_{run_number}'

        # Output directory
        out_dir = f'../output/{run.name}'
        os.makedirs(out_dir, exist_ok=True)

        # load data
        # print('load data')
        datasets, dataloaders, t1_mask = load_cesm2_by_period(data_dir, **config)

        # dimension
        x, y = datasets['train'][0]
        x_shape = tuple(x.shape)
        y_shape = tuple(y.shape)
        n_channels = x.shape[0]

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')
        t1_mask = t1_mask.to(device)

        # Define a model
        model = UNet(
            in_ch=n_channels, 
            out_ch=1, 
            init_ch=config.init_ch, 
            depth=config.depth,
            in_shape=x_shape,
            out_shape=y_shape,
            attention=config.attention, 
            is_res=config.is_res,
            ).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config['total_params'] = total_params

        # Save hyperparameters
        with open(f'{out_dir}/hyperparameters.json', 'w') as f:
            json.dump(dict(config), f)
            
        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
        early_stopper = EarlyStopper(patience=20, min_delta=0.02, out_dir=out_dir)
        trainer = Trainer(
            config.n_epochs, model, device, optimizer, dataloaders,
            t1_mask, early_stopper,
        )
        history = trainer()

        # Save history
        history.astype('float').to_csv(f'{out_dir}/history.csv')

wandb.agent(sweep_id, train_sweep, count=args.count)
