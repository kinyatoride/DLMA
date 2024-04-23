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
        optimizer, dataloaders, t1_wgt,
        early_stopper,
        verbose=True
    ):
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.dataloaders = dataloaders
        self.t1_wgt = t1_wgt
        self.early_stopper = early_stopper
        self.verbose = verbose

    def __call__(self):
        history_list = []
            
        for epoch in range(self.n_epochs):      
            # Training
            self.model.train()
            train_loss = self.test(self.dataloaders['train'])
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = self.test(self.dataloaders['val'], is_train=False)

            # Print loss
            if self.verbose:
                print(f'Epoch {epoch:3d}, '
                      f'train: {train_loss:7.3g}, '
                      f'val: {val_loss:7.3g}'
                     )
            
            # Store loss
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                }
            history_list.append(metrics)
            
            # Log metrics to wandb
            wandb.log(metrics)

            # Save model
            if epoch % 10 == 9:
                self.early_stopper.save_model(self.model.state_dict(), epoch)                         

            # Early stop
            if self.early_stopper(val_loss, epoch, self.model):
                print(f'Early stop at {epoch} epoch')
                break
            
        history = pd.DataFrame(history_list)

        # Save the best model
        self.early_stopper.save_best_model()
        
        return history
            
    def test(self, dataloader, is_train=True):
        loss_epoch = []
        total_size = len(dataloader.dataset)
        
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            batch_size = x.size(0)
                        
            y_pred = self.model(x)
            loss = ((y - y_pred) ** 2 * self.t1_wgt).sum()
            
            if is_train:
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()        
            
            loss_epoch.append(loss)

        loss_epoch = torch.tensor(loss_epoch).sum() / total_size    
        
        return loss_epoch