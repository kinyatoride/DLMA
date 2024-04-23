"""
UNet for different input and output shapes
"""
import torch.nn as nn
import torch.nn.functional as F
from UNet import Encoder, Decoder
    
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, depth=5, init_ch=64, 
                 in_shape=(3, 21, 72), out_shape=(11, 86),
                 attention=True, is_res=True):
        super().__init__()

        self.padding_height = max(out_shape[0] - in_shape[1], 0)
        self.padding_width = max(out_shape[1] - in_shape[2], 0)
        self.crop_height = max(in_shape[1] - out_shape[0], 0)
        self.crop_width = max(in_shape[2] - out_shape[1], 0)

        self.last_ch = init_ch * 2 ** (depth - 1)
        self.encoder = Encoder(in_ch, init_ch, depth, is_res=is_res)
        self.decoder = Decoder(self.last_ch, depth-1, attention, is_res=is_res)
        self.conv = nn.Conv2d(init_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.padding_height > 0 or self.padding_width > 0:
            x = F.pad(x, (
                self.padding_width//2, 
                self.padding_width - self.padding_width//2, 
                self.padding_height//2, 
                self.padding_height - self.padding_height//2
                ))

        x_list = self.encoder(x)
        x = self.decoder(x_list)
        x = self.conv(x)

        if self.crop_height > 0:
            x = x[:, :, :-self.crop_height, :]
        if self.crop_width > 0:
            x = x[:, :, :, :-self.crop_width]

        return x[:, 0]