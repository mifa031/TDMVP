import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

from layers import (ChannelAggregationFFN, MultiOrderGatedAggregation)
from models.model_ltsf import Model_DLinear

class DecomposedPredictor(nn.Module):
    def __init__(self, feat_h, feat_w, feat_c, time_steps=10, time_kernel_size=1, groups=1, expand_ratio=4):
        super().__init__()
        self.feat_area = feat_h * feat_w
        self.feat_c = feat_c
        self.expand_ratio = expand_ratio
        
        if time_steps <= 1:
            self.time_steps = 2
        else:
            self.time_steps = time_steps

        if feat_c % groups != 0:
            groups=1
        
        #trend-remainder decomp prediction
        # [Batch, Input length, Channel] -> [Batch, output length, Channel]
        self.decomp_predictor_1 = Model_DLinear(seq_len=self.time_steps, pred_len=self.time_steps*expand_ratio, individual=False, enc_in=self.feat_area, kernel_size=time_kernel_size)
        self.decomp_predictor_2 = Model_DLinear(seq_len=self.time_steps*expand_ratio, pred_len=self.time_steps, individual=False, enc_in=self.feat_area, kernel_size=time_kernel_size)
        self.dwconv = nn.Conv2d(feat_c*expand_ratio, feat_c*expand_ratio, kernel_size=3, stride=1, padding='same', groups=feat_c*expand_ratio)
        self.conv_spatial_1 = nn.Conv2d(feat_c//self.time_steps, (feat_c//self.time_steps)*expand_ratio, kernel_size=1, stride=1, padding='same', groups=groups)
        self.conv_spatial_2 = nn.Conv2d((feat_c//self.time_steps)*expand_ratio, feat_c//self.time_steps, kernel_size=1, stride=1, padding='same', groups=groups)
        self.act = nn.GELU()
        
        self.decompose = nn.Conv2d(
            in_channels=feat_c*expand_ratio, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(
            1e-5 * torch.ones((1, feat_c*expand_ratio, 1, 1)), requires_grad=True)
        self.decompose_act = nn.GELU()
    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x
        
    def forward(self, x):
        B,C,H,W = x.shape
        
        x_decomp = x.reshape(B, self.time_steps, C//self.time_steps, H, W)
        x_decomp = x_decomp.transpose(1,2)
        x_decomp = x_decomp.reshape(B*(C//self.time_steps), self.time_steps, H*W)
        x_decomp = self.decomp_predictor_1(x_decomp)
        x_decomp = x_decomp.reshape(B, C//self.time_steps, self.time_steps*self.expand_ratio, H, W)
        x_decomp = x_decomp.transpose(1,2)
        x_decomp = x_decomp.reshape(B, C*self.expand_ratio, H, W)

        x_spatio = x.reshape(B, self.time_steps, C//self.time_steps, H, W)
        x_spatio = x_spatio.reshape(B * self.time_steps, C//self.time_steps, H, W)
        x_spatio = self.conv_spatial_1(x_spatio)
        x_spatio = x_spatio.reshape(B, C*self.expand_ratio, H, W)

        x = x_decomp + x_spatio

        x = self.act(self.dwconv(x))
        x = self.feat_decompose(x)

        x_decomp = x.reshape(B, self.time_steps*self.expand_ratio, C//self.time_steps, H, W)
        x_decomp = x_decomp.transpose(1,2)
        x_decomp = x_decomp.reshape(B*(C//self.time_steps), self.time_steps*self.expand_ratio, H*W)
        x_decomp = self.decomp_predictor_2(x_decomp)
        x_decomp = x_decomp.reshape(B, C//self.time_steps, self.time_steps, H, W)
        x_decomp = x_decomp.transpose(1,2)
        x_decomp = x_decomp.reshape(B, C, H, W)

        x_spatio = x.reshape(B, self.time_steps, (C//self.time_steps)*self.expand_ratio, H, W)
        x_spatio = x_spatio.reshape(B * self.time_steps, (C//self.time_steps)*self.expand_ratio, H, W)
        x_spatio = self.conv_spatial_2(x_spatio)
        x_spatio = x_spatio.reshape(B, C, H, W)

        x = x_decomp + x_spatio
        
        return x

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self, embed_dims, out_seq_len, feat_h, feat_w, time_kernel_size, groups=4, mlp_ratio=4., drop_rate=0., drop_path_rate=0., init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        self.out_seq_len = out_seq_len

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # spatial aggregation
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(embed_dims, attn_dw_dilation=attn_dw_dilation)

        # channel aggregation
        self.norm2 = nn.BatchNorm2d(embed_dims)
        #mlp_hidden_dims = int(embed_dims * mlp_ratio)
        #self.predictor = ChannelAggregationFFN(embed_dims=embed_dims, mlp_hidden_dims=mlp_hidden_dims, ffn_drop=drop_rate)
        self.predictor = DecomposedPredictor(feat_h, feat_w, embed_dims, time_steps=out_seq_len, time_kernel_size=time_kernel_size, groups=groups)
        
        # init layer scale
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        B,C,H,W = x.shape #(16 640 16 16)
        T = self.out_seq_len

        # spatial aggregation
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))

        # channel aggregation
        x = x + self.drop_path(self.layer_scale_2 * self.predictor(self.norm2(x)))
        
        x = x.reshape(B, C, H, W)
        return x
