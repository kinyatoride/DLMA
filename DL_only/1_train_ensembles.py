import os
import sys
import json
import torch
import wandb 
import argparse

module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ml_only.loaddata import load_cesm2_by_period
from ml_only.UNet_dynamic import UNet
from ml_only.trainer import Trainer, EarlyStopper
from utils import DotDict

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=1, help='Initial month')
parser.add_argument('--lead', type=int, default=12, help='Targed lead time (months)')
parser.add_argument('--depth', type=int, default=4, help='Number of layers in UNet')
parser.add_argument('--init-ch', type=int, default=256, help='Initial channel size in UNet')
parser.add_argument('--n-epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5, help='Learning rate')

parser.add_argument('--i-ens', type=int, default=0, help='Initial number of ensemble runs')
parser.add_argument('--n-ens', type=int, default=1, help='Number of ensemble runs')
args = parser.parse_args()

# Hyperparameters
hp = {
    'vnames': ['sst', 'ssh', 'taux'],
    'lat_slice': (-50, 50),
    'target_vname': 'sst',
    'target_grid': '2x2',
    'target_lat_slice': (-10, 10),
    'target_lon_slice': (120, 290),
    'lead': args.lead,
    'month': args.month,
    'periods': {
        'train': (1865, 1958),
        'val': (1959, 1985),
        'test': (1986, 1998),
    },
    'batch_size': 16,
    'learning_rate': args.learning_rate,
    'model': 'UNet',
    'attention': False,
    'is_res': False,
    'depth': args.depth,
    'init_ch': args.init_ch,
    'n_epochs': args.n_epochs,
}
hp = DotDict(hp)

# Parameters
vname_join = '_'.join(hp.vnames)

exp_base = f'ml_unet{hp.depth}-{hp.init_ch}_month{hp.month:02d}_lead{hp.lead}_lr{hp.learning_rate:.1e}'

data_dir = '../data/cesm2'

# load data
datasets, dataloaders, t1_wgt = load_cesm2_by_period(data_dir, **hp)

# dimension
x, y = datasets['train'][0]
x_shape = tuple(x.shape)
y_shape = tuple(y.shape)
n_channels = x.shape[0]

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
t1_wgt = t1_wgt.to(device)

for i in range(args.i_ens, args.i_ens + args.n_ens):
    exp = f'{exp_base}_{i}'
    out_dir = f'../output/{exp}'

    # Define a model
    model = UNet(
        in_ch=n_channels, 
        out_ch=1, 
        init_ch=hp.init_ch, 
        depth=hp.depth,
        in_shape=x_shape,
        out_shape=y_shape,
        attention=hp.attention, 
        is_res=hp.is_res,
        ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hp['total_params'] = total_params

    # Compile model (torch 2.x)
    # model = torch.compile(model)

    # Initialize wandb
    wandb.init(
        project='ML-ENSO', config=hp,
        name=exp,
    )

    # Save hyperparameters
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/hyperparameters.json', 'w') as f:
        json.dump(hp, f)
        
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate) 
    early_stopper = EarlyStopper(patience=50, min_delta=0.02, out_dir=out_dir)
    trainer = Trainer(
        hp.n_epochs, model, device, optimizer, dataloaders,
        t1_wgt, early_stopper,
    )
    history = trainer()

    # Save history
    history.astype('float').to_csv(f'{out_dir}/history.csv')

    # Close wandb run
    wandb.finish()
