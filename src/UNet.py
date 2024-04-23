import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_tensor(source, target): 
    """Pad source tensor to match target tensor size"""
    diff_y = target.size()[2] - source.size()[2]
    diff_x = target.size()[3] - source.size()[3]
    
    return F.pad(source, [diff_x // 2, diff_x - diff_x // 2, 
                          diff_y // 2, diff_y - diff_y // 2])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),         
        )
    def forward(self, x):
        return self.conv(x)    

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, is_res=True):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.is_res = is_res
    def forward(self, x):
        x = self.conv1(x)
        if self.is_res:
            identity = x.clone()
        
        x = self.conv2(x)
        
        if self.is_res:
            x = x + identity
            
        return self.relu(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, attention=None, is_res=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention
        self.is_res = is_res
    def forward(self, x, bridge):
        x = self.up(x)
        x = pad_tensor(x, bridge)

        # Attention
        if self.attention:
            bridge = self.attention(bridge, x)
                
        # Concat
        x = torch.cat([x, bridge], dim=1)

        x = self.conv1(x) 
        if self.is_res:
            identity = x.clone()
            
        x = self.conv2(x)
        if self.is_res:
            x = x + identity
            
        return self.relu(x)
    
class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.wx = nn.Conv2d(ch, ch, 1)
        self.wg = nn.Conv2d(ch, ch, 1)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(ch, ch, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, g):
        identity = x.clone()
        x = self.wx(x)
        g = self.wg(g)
        x = self.relu(x + g)
        x = self.psi(x)
        x = self.sigmoid(x)
        return identity * x

class Encoder(nn.Module):
    def __init__(self, in_ch=2, out_ch=64, depth=5, is_res=True):
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList()
        for _ in range(depth):
            self.downs.append(Down(in_ch, out_ch, is_res=is_res))
            in_ch = out_ch
            out_ch *= 2
    def forward(self, x):
        x_list = []
        for i, down in enumerate(self.downs):
            if i > 0:
                x = self.pool(x)
            x = down(x)
            x_list.append(x)
        return x_list
    
class Decoder(nn.Module):
    def __init__(self, in_ch=1024, depth=4, attention=True, is_res=True):
        super().__init__()
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(
                Up(in_ch, in_ch // 2, 
                   Attention(in_ch // 2) if attention else None,
                   is_res=is_res)
                )
            in_ch //= 2
    def forward(self, x_list):
        for i, up in enumerate(self.ups):
            if i == 0:
                x = x_list[-1]
            bridge = x_list[-2-i]
            x = up(x, bridge)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, depth=5, init_ch=64, 
                 attention=True, is_res=True):
        super().__init__()
        self.last_ch = init_ch * 2 ** (depth - 1)
        self.encoder = Encoder(in_ch, init_ch, depth, is_res=is_res)
        self.decoder = Decoder(self.last_ch, depth-1, attention, is_res=is_res)
        self.conv = nn.Conv2d(init_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_list = self.encoder(x)
        x = self.decoder(x_list)
        x = self.conv(x)
        return self.sigmoid(x)