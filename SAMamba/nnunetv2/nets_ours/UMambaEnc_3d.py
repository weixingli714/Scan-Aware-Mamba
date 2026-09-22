from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He


# =========================================================
# 基础 Mamba 序列模块
# 输入 : [B, L, C]
# 输出 : [B, L, C]
# =========================================================
class MambaSeq(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mamba(x)
        return x


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# =========================================================
# V2 风格 GSC
# GSC(z) = z + Conv3x3( Conv3x3(z) * Conv1x1(z) )
# =========================================================
class ConvNormAct(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GSCV2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3_a = ConvNormAct(in_channels, kernel_size=3, padding=1)
        self.conv3x3_b = ConvNormAct(in_channels, kernel_size=3, padding=1)
        self.conv1x1 = ConvNormAct(in_channels, kernel_size=1, padding=0)
        self.fuse3x3 = ConvNormAct(in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        feat_3 = self.conv3x3_a(x)
        feat_3 = self.conv3x3_b(feat_3)
        feat_1 = self.conv1x1(x)
        gated = feat_3 * feat_1
        out = self.fuse3x3(gated)
        return out + residual


# =========================================================
# 对 [B, C, D, W, H] 做 LayerNorm(C)
# =========================================================
class LayerNormChannelLast3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = rearrange(x, 'b c d w h -> b d w h c')
        x = self.norm(x)
        x = rearrange(x, 'b d w h c -> b c d w h')
        return x


# =========================================================
# 单个 view 下的 ToM
# 输入统一抽象为 [B, C, G, X, Y]
# =========================================================
class ToM2DViewV2(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.seq_forward = MambaSeq(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.seq_reverse = MambaSeq(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.seq_cross = MambaSeq(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def build_forward_seq(self, x):
        # [B, C, G, X, Y] -> [B, G*X*Y, C]
        return rearrange(x, 'b c g x y -> b (g x y) c')

    def build_reverse_seq(self, x):
        seq = self.build_forward_seq(x)
        seq = torch.flip(seq, dims=[1])
        return seq

    def build_cross_group_seq(self, x):
        # [B, C, G, X, Y] -> [B, X*Y*G, C]
        return rearrange(x, 'b c g x y -> b (x y g) c')

    def seq_to_view(self, seq, G, X, Y):
        return rearrange(seq, 'b (g x y) c -> b c g x y', g=G, x=X, y=Y)

    def forward(self, x):
        # x: [B, C, G, X, Y]
        B, C, G, X, Y = x.shape

        seq_f = self.build_forward_seq(x)
        out_f = self.seq_forward(seq_f)
        out_f = self.seq_to_view(out_f, G, X, Y)

        seq_r = self.build_reverse_seq(x)
        out_r = self.seq_reverse(seq_r)
        out_r = torch.flip(out_r, dims=[1])
        out_r = self.seq_to_view(out_r, G, X, Y)

        seq_c = self.build_cross_group_seq(x)
        out_c = self.seq_cross(seq_c)
        out_c = rearrange(out_c, 'b (x y g) c -> b c g x y', g=G, x=X, y=Y)

        out = out_f + out_r + out_c
        return out


# =========================================================
# ToOM：三个正交平面
# 输入 / 输出: [B, C, D, W, H]
# =========================================================
class ToOMV2Layer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.tom_axial = ToM2DViewV2(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.tom_coronal = ToM2DViewV2(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.tom_sagittal = ToM2DViewV2(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: [B, C, D, W, H]

        # Axial: G=D, X=W, Y=H
        out_axial = self.tom_axial(x)

        # Coronal: 固定 W，看 D x H
        x_coronal = rearrange(x, 'b c d w h -> b c w d h')
        out_coronal = self.tom_coronal(x_coronal)
        out_coronal = rearrange(out_coronal, 'b c w d h -> b c d w h')

        # Sagittal: 固定 H，看 D x W
        x_sagittal = rearrange(x, 'b c d w h -> b c h d w')
        out_sagittal = self.tom_sagittal(x_sagittal)
        out_sagittal = rearrange(out_sagittal, 'b c h d w -> b c d w h')

        return out_axial + out_coronal + out_sagittal


# =========================================================
# TSMamba Block
# z_hat   = GSC(z)
# z_tilde = ToOM(LN(z_hat)) + z_hat
# z_out   = MLP(LN(z_tilde)) + z_tilde
# =========================================================
class TSMambaBlockV2(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.gsc = GSCV2(dim)
        self.norm1 = LayerNormChannelLast3D(dim)
        self.toom = ToOMV2Layer(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = LayerNormChannelLast3D(dim)
        self.mlp = MlpChannel(dim, int(dim * mlp_ratio))

    def forward(self, x):
        z_hat = self.gsc(x)
        z_tilde = self.toom(self.norm1(z_hat)) + z_hat
        z_out = self.mlp(self.norm2(z_tilde)) + z_tilde
        return z_out


# =========================================================
# 大核卷积模块
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class LargeKernelConv(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.deep_path = nn.Sequential(
            ConvBlock(channels, channels, 5, padding=2),
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 3, padding=2, dilation=2),
        )

        self.shortcut_path = nn.Sequential(
            ConvBlock(channels, channels, 3, padding=1),
            ConvBlock(channels, channels, 1, padding=0)
        )

        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.deep_path(x) + self.shortcut_path(x) + x
        return self.se(out)


# =========================================================
# SegMamba Encoder
# 输入是 encoder1 的输出，因此输入通道 = feat_size[0]
# 输出 4 个尺度特征
# =========================================================
class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=48,
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()

        assert len(depths) == 4, "depths must have length 4"
        assert len(dims) == 4, "dims must have length 4"

        self.downsample_layers = nn.ModuleList()

        # stage 0
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(dims[0]),
            nn.ReLU(inplace=True),
        )
        self.downsample_layers.append(stem)

        # stage 1~3
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(dims[i + 1]),
                nn.ReLU(inplace=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()

        for i in range(4):
            gsc = nn.Sequential(*[LargeKernelConv(dims[i]) for _ in range(depths[i])])

            # 前两层只做卷积增强；后两层加入 TSMambaBlockV2
            if i < 2:
                stage = nn.Identity()
            else:
                stage = nn.Sequential(*[TSMambaBlockV2(dim=dims[i], mlp_ratio=2.0) for _ in range(depths[i])])

            self.gscs.append(gsc)
            self.stages.append(stage)

        self.out_indices = out_indices

        self.norms = nn.ModuleList([nn.InstanceNorm3d(dims[i]) for i in range(4)])
        self.mlps = nn.ModuleList([MlpChannel(dims[i], 2 * dims[i]) for i in range(4)])

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                x_out = self.norms[i](x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


# =========================================================
# nnUNet 风格封装后的 SegMamba
# 输入输出接口兼容 nnUNet
# =========================================================
class SegMambaNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        feat_size: Union[List[int], Tuple[int, ...]] = (48, 96, 192, 384),
        depths: Union[List[int], Tuple[int, ...]] = (2, 2, 2, 2),
        hidden_size: int = None,
        deep_supervision: bool = True,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        if hidden_size is None:
            hidden_size = feat_size[-1] * 2

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feat_size = list(feat_size)
        self.depths = list(depths)
        self.hidden_size = hidden_size
        self.deep_supervision = deep_supervision
        self.spatial_dims = spatial_dims

        # encoder1: 原图 -> feat_size[0]
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.input_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # SegMamba 主干
        self.vit = MambaEncoder(
            in_chans=self.feat_size[0],
            depths=self.depths,
            dims=self.feat_size,
            out_indices=(0, 1, 2, 3),
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 主输出
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.num_classes,
        )

        # Deep Supervision 输出头
        if self.deep_supervision:
            self.ds_out_1 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.num_classes,
            )  # dec1, 1/2
            self.ds_out_2 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.num_classes,
            )  # dec2, 1/4
            self.ds_out_3 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.num_classes,
            )  # dec3, 1/8

    def forward(self, x_in):
        # full-res skip
        enc1 = self.encoder1(x_in)

        # multi-scale features from SegMamba encoder
        outs = self.vit(enc1)
        x2, x3, x4, x5 = outs

        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(x5)

        dec3 = self.decoder5(enc_hidden, enc4)  # 1/8
        dec2 = self.decoder4(dec3, enc3)        # 1/4
        dec1 = self.decoder3(dec2, enc2)        # 1/2
        dec0 = self.decoder2(dec1, enc1)        # full
        out = self.decoder1(dec0)               # full

        out_main = self.out(out)

        if not self.deep_supervision:
            return out_main

        out_ds1 = self.ds_out_1(dec1)
        out_ds2 = self.ds_out_2(dec2)
        out_ds3 = self.ds_out_3(dec3)

        # nnUNet v2 通常期望高分辨率在前
        return [out_main, out_ds1, out_ds2, out_ds3]


# =========================================================
# 从 nnUNet v2 plans 构建 SegMamba
# =========================================================
def get_segmamba_3d_from_plans(
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    deep_supervision: bool = True
):
    """
    按 nnUNet v2 的 plans/configuration 构建 SegMamba 网络
    """

    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    assert dim == 3, "This implementation only supports 3D."

    conv_op = convert_dim_to_conv_op(dim)
    _ = get_matching_instancenorm(conv_op)  # 保留风格一致性

    label_manager = plans_manager.get_label_manager(dataset_json)

    # 这里沿用 nnUNet 的 feature 生成规则
    features_per_stage = [
        min(configuration_manager.UNet_base_num_features * 2 ** i,
            configuration_manager.unet_max_num_features)
        for i in range(num_stages)
    ]

    # SegMamba 原始结构是 4-stage，因此这里取前 4 个 stage
    # 如果 plans 超过 4 个 stage，就只取前 4 个
    if len(features_per_stage) < 4:
        raise ValueError(
            f"SegMambaNet expects at least 4 stages from plans, but got {len(features_per_stage)}."
        )

    feat_size = features_per_stage[:4]

    # depths 这里先固定成比较常见的 2,2,2,2
    # 你后续也可以改成 [1,1,1,1] 或 [2,2,2,2]
    depths = (2, 2, 2, 2)

    model = SegMambaNet(
        input_channels=num_input_channels,
        num_classes=label_manager.num_segmentation_heads,
        feat_size=feat_size,
        depths=depths,
        hidden_size=feat_size[-1] * 2,
        deep_supervision=deep_supervision,
        norm_name="instance",
        res_block=True,
        spatial_dims=3,
    )

    model.apply(InitWeights_He(1e-2))
    return model