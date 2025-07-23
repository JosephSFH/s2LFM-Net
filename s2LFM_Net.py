import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import math

class s2lfm_Net(nn.Module):
    """
    Main network for s2LFM-Net.
    Processes input RGB light field images and outputs multiplex results with full angular resolution.
    """
    def __init__(self, angular_in, out_channels=4):
        super(s2lfm_Net, self).__init__()
        channel = 64
        self.angRes = angular_in
        self.out_channels = out_channels

        # Spectral-angular fusion module
        self.FeaExtract = InitFeaExtract(channel)  # Initial feature extraction

        # Spectral-spatial extraction module
        self.D3Unet = UNet(channel, channel, channel)  # 3D U-Net for feature learning

        # Spatial-angular interaction module
        self.Out = nn.Conv2d(in_channels=channel, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Angular_UpSample = Upsample(channel, angular_in)

        # Residual
        self.Resup = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, x):
        """
        Forward pass for the network.
        """
        Conv_input = self.Resup(x)
        Conv_cat = LFsplit(Conv_input, self.angRes)
        x_mv = LFsplit(x, self.angRes)
        b, n, c, h, w = x_mv.shape

        buffer_mv_initial = self.FeaExtract(x_mv)
        buffer_mv = self.D3Unet(buffer_mv_initial.permute(0,2,1,3,4))  # 3D U-Net expects different dimension order
        HAR = self.Angular_UpSample(buffer_mv)
        out = self.Out(HAR.contiguous().view(b*self.angRes*self.angRes, -1, h, w))
        # Combine network output with shortcut connection
        out = FormationCheck(out.contiguous().view(b,-1, self.out_channels, h, w)) + FormationCheck(Conv_cat)

        return out

class Upsample(nn.Module):
    """
    Spatial-angular interaction module.
    """
    def __init__(self, channel, angular_in):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.an_out = angular_in
        self.angconv = nn.Sequential(
                        nn.Conv2d(in_channels=channel*2, out_channels=channel*2, kernel_size=3, padding=1, bias=False),
                        nn.LeakyReLU(0.1, inplace=True))
        self.upsp = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(1),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        """
        Forward pass for feature interaction.
        """
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)

        out = out.view(b,h*w,-1,self.an_out*self.an_out)
        out = torch.transpose(out,1,3)
        out = out.contiguous().view(b, self.an_out*self.an_out, -1, h, w)
        return out
    
class D3Resblock(nn.Module):
    """
    3D Residual Block for feature extraction.
    """
    def __init__(self, channel):
        super(D3Resblock, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1), bias=False), 
                                nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1,1,1), bias=False)

    def __call__(self, x_init):
        x = self.conv(x_init)
        x = self.conv_2(x)
        return x + x_init

class SEGating(nn.Module):
    """
    Squeeze-and-Excitation gating for 3D features.
    """
    def __init__(self , inplanes , reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes , inplanes , kernel_size=1 , stride=1 , bias=True),
            nn.Sigmoid()
        )
        
    def forward(self , x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y

class UNet(nn.Module):
    """
    3D U-Net architecture for feature extraction and fusion.
    """
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Downsampling path
        self.down_1 = D3Resblock(self.in_dim)
        self.pool_1 = stride_conv_3d(self.num_filters, self.num_filters*2, activation)
        self.down_2 = D3Resblock(self.num_filters*2)
        self.pool_2 = stride_conv_3d(self.num_filters * 2, self.num_filters * 3, activation)
        
        # Bridge
        self.bridge_1 = D3Resblock(self.num_filters * 3)
               
        # Upsampling path
        self.trans_1 = conv_trans_block_3d(self.num_filters * 3, self.num_filters * 2, activation)
        self.up_1 = D3Resblock(self.num_filters * 2)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_2 = D3Resblock(self.num_filters * 1)
        
        self.out_2D = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        """
        Forward pass for U-Net.
        """
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)        
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        
        bridge = self.bridge_1(pool_2)
           
        trans_1 = self.trans_1(bridge)
        addition_1 = trans_1 + down_2
        up_1 = self.up_1(addition_1)        
        trans_2 = self.trans_2(up_1)
        addition_2 = trans_2 + down_1
        up_2 = self.up_2(addition_2)

        # Concatenate skip connection and output
        out = torch.cat((up_2, x), 1).permute(0,2,1,3,4)
        b,n, c,h,w = out.shape
        out = self.out_2D(out.contiguous().view(b*n, c, h, w)).view(b, n, c, h, w)
        return out
    
def conv_block_3d(in_dim, out_dim, activation):
    """
    3D convolutional block with activation.
    """
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        activation)

def stride_conv_3d(in_dim, out_dim, activation):
    """
    3D convolutional block with stride for downsampling.
    """
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
        activation)

def conv_trans_block_3d(in_dim, out_dim, activation):
    """
    3D transposed convolutional block for upsampling.
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1), bias=False),
        activation)

class InitFeaExtract(nn.Module):
    """
    Initial feature extraction from input images.
    """
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, r, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        buffer = self.FEconv(x)
        _, c, h, w = buffer.shape
        buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h, w)
        return buffer

def LFsplit(data, angRes):
    """
    Split input tensor into sub-aperture images (views) for light field processing.
    """
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st

def FormationCheck(x_sv):
    """
    Reconstructs the light field image from its split views.
    """
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)
    return out
