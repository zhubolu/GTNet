import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import _load_checkpoint

class AMSELayer(nn.Module):
    def __init__(self,channel,reduction=4):
        super(AMSELayer,self).__init__()
        "全局平均池化和全局最大池化组成的amse"
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.gmp = nn.AdaptiveAvgPool2d(1)
        # self.linar_a = nn.Sequential(
        #     nn.Conv2d(channel,channel//4,1,bias=False),
        #     nn.BatchNorm2d(channel//4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel//4, 16 , 1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )
        # self.linar_m = nn.Sequential(
        #     nn.Conv2d(channel, channel // 4, 1, bias=False),
        #     nn.BatchNorm2d(channel // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // 4, 16, 1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),

        )
        self.active = nn.Sigmoid()

    def forward(self,x):
        b,c,_,_ = x.size()
        x_a = self.gap(x)


        y = torch.bmm(x_a.view(b,c,1),x_a.view(b,1,c)) # b c  c
        # b c c ---> b c 1 1

        y = torch.sum(y,dim=1)

        y = self.fc(y)

        y = self.active((y.view(b,c,1,1)+x_a))
        return x * y.expand_as(x)
class Pool_transform(nn.Module):
    def __init__(self,in_channels):
        super(Pool_transform,self).__init__()
        self.linear_transform1 = nn.Sequential(nn.Conv2d(in_channels, in_channels//4, 1, bias=False),
                     nn.BatchNorm2d(in_channels//4), nn.ReLU(inplace=True),
                     nn.Conv2d(in_channels//4, in_channels, 1, bias=False),
                 nn.BatchNorm2d(in_channels ), nn.ReLU(inplace=True)
                      )
        self.linear_transform2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                                               nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
                                               nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
                                               nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)
                                               )

    def forward(self,pool_a,pool_m):
        b,c,n = pool_a.size()
        #增加维度 4个维度才能使用卷积
        x_q = self.linear_transform1(pool_a[:,:,:,None]) # b c n 1
        x_k = self.linear_transform2(pool_m[:,:,:,None]).view(b,c,n).permute(0,2,1) # b n c
        out = torch.bmm(x_q.view(b,c,n),x_k) # b c c
        return out
class _PSPModule(nn.Module):
    def __init__(self, in_channels,out_channels ,bin_sizes=[1,2,3,6],model='avg'):
        super(_PSPModule, self).__init__()
        self.model = model
        self.out_chanels = out_channels
        "池化并调整通道"
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                      for b_s in bin_sizes])

    def _make_stages(self, in_channels, out_channels, bin_sz):
        if self.model =='avg':
            prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        else:
            prior = nn.AdaptiveMaxPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        b,c,h, w = features.size()
        pyramids = []
        pyramids.extend([stage(features).view(b,self.out_chanels,-1) for stage in self.stages])
        out = torch.cat(pyramids,dim=2)

        return out
class Head3(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, h=64,softmax=False):
        super(Head3, self).__init__()
        self.h = h
        self.sof = softmax
        self.query =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.key =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.linear_h = nn.Sequential(
            nn.Conv2d(out_channels,out_channels//reduction,1,bias=False),
            nn.BatchNorm2d(out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels //reduction,out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #
        )
        self.linear_w = nn.Sequential(
            nn.Conv2d(out_channels,out_channels//reduction,1,bias=False),
            nn.BatchNorm2d(out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//reduction,out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)

        b, c, h, w = query.size()
        #print(b, c, h, w)
        q_h = self.pool_h(query)
        k_w = self.pool_w(key)

        q_h = self.linear_h(q_h)  # b c w 1
        k_w = self.linear_w(k_w)  # b c 1 h

        q_h = q_h.view(b * c, w, -1)
        k_w = k_w.view(b * c, -1, h)

        energy = torch.bmm(q_h, k_w).view(b, c, h, w)  # b c h w
        if self.sof:
            attention = self.softmax(energy)
            out = attention
        else:
        #out = torch.mul(attention, value)                   .view(b, c, h, w)
        #out = self.gamma * out + x
            out =energy
        return out
class PF(nn.Module):
    def __init__(self,in_channels=512,out_channels=128):
        super(PF,self).__init__()
        self.head1 = Head3(in_channels, in_channels//2, reduction=4, h=64)  # 这个数等于输入图片数/8
        self.head2 = Head3(in_channels//2, out_channels, reduction=4, h=64, softmax=True)

        self.value = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True), )
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True), )
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        x_ = self.head1(x)
        x_qk = self.head2(x_)
        x_value = self.value(x)
        x_att = torch.mul(x_qk, x_value)  # 计算attention
        x_rdu = self.reduce(x)  # x原始降维不变
        out = self.gamma * x_att + x_rdu
        return out

class CF(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CF,self).__init__()
        self.psp_a = _PSPModule(in_channels,out_channels)
        self.psp_m = _PSPModule(in_channels,out_channels) #,model='max'
        self.qk = Pool_transform(out_channels)
        self.value = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True), )
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True), )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        x_pool_a = self.psp_a(x)
        x_pool_m = self.psp_m(x)
        x_qk = self.qk(x_pool_a,x_pool_m)
        x_value = self.value(x)
        b,c,h,w = x_value.size()
        x_rdu = self.reduce(x)
        x_att = self.softmax((torch.bmm(x_qk,x_value.view(b,-1,h*w))).view(b,c,h,w)) #
        out = self.gamma * x_att + x_rdu
        return out
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            activations=None,
            norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Module):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
                activation()
            )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x


class GroupFormer(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN', requires_grad=True), ):


        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.conv1=Conv2d_BN(dim,dim,1,norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()


    def forward(self, x):  # x (B,N,C)
        split_tensors = torch.split(x, split_size_or_sections=x.size(1) // 2, dim=1)

        # 将分割后的两部分分别赋值给 X1 和 X2
        X1, X2 = split_tensors
        B, C, H, W = X1.shape

        q = self.to_q(X1)
        k = self.to_k(X1)
        v = self.to_v(X1)


        # detail enhance

        qkv = self.act(self.conv1(X2))


        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx)
        xx=torch.cat([xx,qkv], dim=1)
        return xx


class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = GroupFormer(128, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(self, block_num=4, embedding_dim=256, key_dim=16, num_heads=8,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel]*c2*c1).reshape(c2,c1,3,3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):   # 构建 CSP Bottleneck 结构
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
class TPM(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self):
        super(TPM, self).__init__()
        self.MP1 = nn.MaxPool2d(2,2)
        self.MP2 = nn.MaxPool2d(4,4)
        self.conv1=Conv(896,512,1,1)



    def forward(self, x):

        l1=x[0]
        l2=x[1]
        l3=x[2]
        l2=self.MP1(l2)
        l3=self.MP2(l3)
        l=torch.cat((l1,l2,l3),dim=1)
        l=self.conv1(l)
        return l



class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)


    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=13, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv

            bs, _, ny, nx = x[i].shape
            x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]

                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]


                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):

        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


