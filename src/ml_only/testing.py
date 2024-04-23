import os
import torch
import numpy as np
from loss import ma_loss, ma_fcst_err

def test(
    model, device, dataloader, t1_wgt,
):    
    model.eval()
    loss_list = []
    y_pred_list = []

    total_size = len(dataloader.dataset)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
                        
            y_pred = model(x)
            loss = ((y - y_pred) ** 2 * t1_wgt).sum()
            
            loss_list.append(loss)
            y_pred_list.append(y_pred)

    loss = torch.tensor(loss_list).sum().item() / total_size
    y_pred = torch.cat(y_pred_list, dim=0)
    
    return loss, y_pred