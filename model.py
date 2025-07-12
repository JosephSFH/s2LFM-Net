import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import math

class s2lfm_Net(nn.Module):
    def __init__(self, angular_in, out_channels=4):
        super(s2lfm_Net, self).__init__()
        channel = 64
        self.angRes = angular_in
        self.out_channels = out_channels
        self.FeaExtract = InitFeaExtract(channel)
        self.D3Unet = UNet(channel, channel, channel)
        self.Out = nn.Conv2d(in_channels=channel, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Angular_UpSample = Upsample(channel, angular_in)
        self.Resup = nn.Conv2d(in_channels=3,out_channels=out_channels,kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, x):
        Conv_input = self.Resup(x)
        Conv_cat = LFsplit(Conv_input,self.angRes)
        x_mv = LFsplit(x, self.angRes)
        b, n, c, h, w = x_mv.shape

        buffer_mv_initial = self.FeaExtract(x_mv)
        buffer_mv = self.D3Unet(buffer_mv_initial.permute(0,2,1,3,4))
        HAR = self.Angular_UpSample(buffer_mv)
        out = self.Out(HAR.contiguous().view(b*self.angRes*self.angRes, -1, h, w))
        out = FormOutput(out.contiguous().view(b,-1, self.out_channels, h, w)) + FormOutput(Conv_cat)

        return out
class Conv2d_refpad(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(Conv2d_refpad, self).__init__()
        pad = kernel//2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, padding=0, bias=False)
 
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
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
class Interpolation(nn.Module):
    def __init__(self, angular_in, factor):
        super(Interpolation, self).__init__()
        self.an = angular_in
        self.an_out = angular_in*factor
        self.factor = factor
    def forward(self, x_mv):
        b, n, c, h, w = x_mv.shape
        x = x_mv.contiguous().view(b, n, c, h*w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b*h*w, c, self.an, self.an)

        out = functional.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)

        out = out.view(b,h*w,c,self.an_out*self.an_out)
        out = torch.transpose(out,1,3)
        out = out.contiguous().view(b, self.an_out*self.an_out, c, h, w) 
        return out
    
class D3Resblock(nn.Module):
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
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.1, inplace=True)
        
        
        self.down_1 = D3Resblock(self.in_dim)
        self.pool_1 = stride_conv_3d(self.num_filters, self.num_filters*2, activation)
        self.down_2 = D3Resblock(self.num_filters*2)
        self.pool_2 = stride_conv_3d(self.num_filters * 2, self.num_filters * 3, activation)
        
        self.bridge_1 = D3Resblock(self.num_filters * 3)
               
        self.trans_1 = conv_trans_block_3d(self.num_filters * 3, self.num_filters * 2, activation)
        self.up_1 = D3Resblock(self.num_filters * 2)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_2 = D3Resblock(self.num_filters * 1)
        
        self.out_2D = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

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

        out = torch.cat((up_2, x), 1).permute(0,2,1,3,4)
        b,n, c,h,w = out.shape
        out = self.out_2D(out.contiguous().view(b*n, c, h, w)).view(b, n, c, h, w)
        return out
    
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        activation)

def stride_conv_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
        activation)

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1), bias=False),
        activation)

class InitFeaExtract(nn.Module):
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
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(x_sv):
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
