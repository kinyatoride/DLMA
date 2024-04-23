import copy
import torch
import wandb
import numpy as np
import pandas as pd
from loss import ma_loss, ma_fcst_err

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_epoch=15, out_dir=None):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epoch = min_epoch
        self.counter = 0
        self.best_loss = np.inf
        self.best_epoch = None
        self.best_model = None
        self.out_dir = out_dir

    def __call__(self, val_loss, epoch, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model.state_dict())

            self.wandb_summary()
            # if epoch > self.min_epoch:
            #     self.save_model()
            
        elif val_loss > (self.best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def wandb_summary(self):
        wandb.run.summary['best_mse'] = self.best_loss
        wandb.run.summary['best_epoch'] = self.best_epoch

    def save_model(self, model, epoch):
        if self.out_dir is not None:
            outf = f'{self.out_dir}/model_epoch{epoch}.pt'
            torch.save(model, outf)

    def save_best_model(self):
        if self.out_dir is not None:
            outf = f'{self.out_dir}/model_epoch{self.best_epoch}.pt'
            torch.save(self.best_model, outf)

class Trainer:
    def __init__(
        self, n_epochs, model, device,
        optimizer, dataloaders, 
        t0_library, t0_mask, n_sub,
        t1_library, n_analog,
        early_stopper,
        verbose=True
    ):
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloaders = dataloaders
        self.t0_library = t0_library
        self.t0_mask = t0_mask
        self.n_sub = n_sub
        self.t1_library = t1_library
        self.n_analog = n_analog
        self.early_stopper = early_stopper
        self.verbose = verbose

    def __call__(self):
        history_list = []
            
        for epoch in range(self.n_epochs):      
            # Training
            self.model.train()
            train_loss, _, train_mse = self.test(self.dataloaders['train'])
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss, _, val_mse = self.test(self.dataloaders['val'], is_train=False)

            # Print loss
            if self.verbose:
                print(f'Epoch {epoch:3d}, '
                      f'train: {train_loss:7.3g}, {train_mse:7.3f}, '
                      f'val: {val_loss:7.3g}, {val_mse:7.3f}'
                     )
            
            # Store loss
            metrics = {
                'train_loss': train_loss,
                'train_mse': train_mse,
                'val_loss': val_loss,
                'val_mse': val_mse,
                }
            history_list.append(metrics)
            
            # Log metrics to wandb
            wandb.log(metrics)

            # Save model
            if epoch % 10 == 9:
                self.early_stopper.save_model(self.model.state_dict(), epoch)                         

            # Early stop
            if self.early_stopper(val_mse, epoch, self.model):
                print(f'Early stop at {epoch} epoch')
                break
            
        history = pd.DataFrame(history_list)

        # Save the best model
        self.early_stopper.save_best_model()
        
        return history
            
    def test(self, dataloader, is_train=True):
        loss_epoch = []
        mean_mse_epoch = []
        mse_epoch = []
        total_size = len(dataloader.dataset)
        
        for x0, x1, t1_dist in dataloader:
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            t1_dist = t1_dist.to(self.device)
            batch_size = x0.size(0)
                        
            weight = self.model(x0)
            weight = torch.where(self.t0_mask[None], 0, weight)
            
            loss, mean_mse, indices = ma_loss(
                x0, self.t0_library, weight, t1_dist, 
                self.n_sub, n_analog=self.n_analog, 
                insample=is_train, return_index=True)
            
            mse = ma_fcst_err(
                x1, self.t1_library, indices, n_analog=self.n_analog)
            
            if is_train:
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()        
            
            loss_epoch.append(loss)
            mean_mse_epoch.append(mean_mse * batch_size)
            mse_epoch.append(mse * batch_size)
        loss_epoch = torch.tensor(loss_epoch).mean() 
        mean_mse_epoch = torch.tensor(mean_mse_epoch).sum() / total_size
        mse_epoch = torch.tensor(mse_epoch).sum() / total_size       
        
        return loss_epoch, mean_mse_epoch, mse_epoch        