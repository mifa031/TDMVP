import torch
from torch import nn

from tdmvp_modules import (ConvSC, MogaSubBlock)
from layers import (ChannelAggregationFFN, MultiOrderGatedAggregation)
from timm.models.layers import DropPath, trunc_normal_
from models.model_ltsf import Model_DLinear
import math

class TDMVP_Model(nn.Module):
    def __init__(self, in_shape, out_seq_len, time_kernel_size=1, groups=4, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='tau',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, **kwargs):
        super(TDMVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        feat_H, feat_W = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))  # downsample 1 / 2**(N_S/2)

        self.prev_seq_len = T
        self.out_seq_len = out_seq_len
        self.out_hid_S = (T * hid_S) // out_seq_len
        self.hid_S = hid_S

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc)
        self.dec = Decoder((T*hid_S)//self.out_seq_len, C, N_S, T, out_seq_len, spatio_kernel_dec)

        model_type = 'moga'
        self.hid = MidMetaNet(T * hid_S, hid_T, N_T, out_seq_len, feat_H, feat_W, time_kernel_size, groups=groups,
                              input_resolution=(H, W), model_type=model_type,
                              mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.reshape(B, self.out_seq_len, (T*C_)//self.out_seq_len, H_, W_) #consider out_seq_len
        B, out_T, mid_C, mid_H, mid_W = z.shape
        hid = self.hid(z)
        hid = hid.reshape(B * out_T, mid_C, mid_H, mid_W)

        Y = self.dec(hid, skip)

        Y = Y.reshape(B, out_T, C, H, W)
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC( C_in, C_hid, spatio_kernel, downsampling=samplings[0]),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, prev_seq_len, out_seq_len, spatio_kernel):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1])
        )
        self.read_out = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        hid_B, hid_C, hid_H, hid_W = hid.shape
        enc1 = enc1.reshape(hid_B, hid_C, hid_H, hid_W)
        Y = self.dec[-1](hid + enc1)
        Y = self.read_out(Y)
        return Y

class MetaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_seq_len, feat_H, feat_W, time_kernel_size, groups=4, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = 'moga'

        if model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, out_seq_len, feat_H, feat_W, time_kernel_size, groups=groups, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, out_seq_len, feat_H, feat_W, time_kernel_size, groups=4,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, out_seq_len, feat_H, feat_W, time_kernel_size, groups, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, out_seq_len, feat_H, feat_W, time_kernel_size, groups, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, out_seq_len, feat_H, feat_W, time_kernel_size, groups, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        x = x.reshape(B, T*C, H, W)
        
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        
        y = z.reshape(B, T, C, H, W)
        return y
