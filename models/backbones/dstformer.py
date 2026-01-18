import torch
import torch.nn as nn
import math
import warnings
import random
import numpy as np
from collections import OrderedDict
from functools import partial
from itertools import repeat
from motionbert.lib.model.drop import DropPath

# --- 基础工具函数 ---
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# --- 基础组件 ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- 核心改造 1: Attention 层 ---
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # 🌟 强感知改造：接收 view_idx 参数
    def forward(self, x, seqlen=1, view_idx=None):
        B, N, C = x.shape

        # 🌟 关键：检查 self.qkv 是否为视角感知 LoRA 层
        # 如果是，则传入 view_idx 进行特征动态调制
        if hasattr(self.qkv, 'view_modulator'):
            qkv = self.qkv(x, view_idx=view_idx)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode == 'spatial':
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'temporal':
            x = self.forward_temporal(q, k, v, seqlen=seqlen)
        else:
            x = self.forward_spatial(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seqlen=8):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ vt).permute(0, 3, 2, 1, 4).reshape(B, N, C * self.num_heads)
        return x

# --- 核心改造 2: Block 层 ---
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, st_mode="spatial")
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, st_mode="temporal")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        self.mlp_s = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    # 🌟 强感知改造：接收 view_idx 并向下传递
    def forward(self, x, seqlen=1, view_idx=None):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen, view_idx=view_idx))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        return x

# --- 核心改造 3: DSTformer 类 ---
class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i], st_mode="stage_st")
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i], st_mode="stage_ts")
            for i in range(depth)])
        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(dim_feat, dim_rep)), ('act', nn.Tanh())])) if dim_rep else nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    # 🌟 强感知改造：透传 view_idx
    def get_representation(self, x, view_idx=None):
        return self.forward(x, return_rep=True, view_idx=view_idx)

    # 🌟 强感知改造：forward 接收并层层透传 view_idx
    def forward(self, x, return_rep=False, view_idx=None):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = self.joints_embed(x) + self.pos_embed
        x = x.reshape(-1, F, J, self.dim_feat) + self.temp_embed[:, :F, :, :]
        x = self.pos_drop(x.reshape(BF, J, -1))

        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            # 🌟 核心：将视角接力棒传给 Block
            x_st = blk_st(x, F, view_idx=view_idx)
            x_ts = blk_ts(x, F, view_idx=view_idx)
            if self.att_fuse:
                alpha = self.ts_attn[idx](torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5

        x = self.norm(x).reshape(B, F, J, -1)
        x = self.pre_logits(x)
        return x if return_rep else self.head(x)