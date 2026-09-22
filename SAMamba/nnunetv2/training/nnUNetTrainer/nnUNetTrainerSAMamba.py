from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision

# =========================================================
# Basic Mamba sequence module
# Input : [B, L, C]
# Output: [B, L, C]
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


# =========================================================
# MLP channel block
# =========================================================
class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()

        self.fc1 = nn.Conv3d(
            hidden_size,
            mlp_dim,
            kernel_size=1,
        )

        self.act = nn.GELU()

        self.fc2 = nn.Conv3d(
            mlp_dim,
            hidden_size,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# =========================================================
# LayerNorm over channel dimension for [B, C, D, W, H]
# =========================================================
class LayerNormChannelLast3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = rearrange(x, "b c d w h -> b d w h c")
        x = self.norm(x)
        x = rearrange(x, "b d w h c -> b c d w h")
        return x


# =========================================================
# Norm helper
# =========================================================
def get_norm_3d(norm_name: str, channels: int):
    if norm_name == "instance":
        return nn.InstanceNorm3d(channels)
    elif norm_name == "batch":
        return nn.BatchNorm3d(channels)
    else:
        raise ValueError(f"Unsupported norm_name: {norm_name}")


# =========================================================
# General Conv + Norm + Act
# =========================================================
class ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        norm_name="instance",
        activation=True,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            get_norm_3d(norm_name, out_channels),
        ]

        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =========================================================
# Channel reduction block:
#   1x1x1 conv + Norm + ReLU
#   3x3x3 conv + Norm + ReLU
# =========================================================
class ChannelReductionBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_name="instance",
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvNormAct3D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_name=norm_name,
                activation=True,
            ),
            ConvNormAct3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_name=norm_name,
                activation=True,
            ),
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# LargeKernelConv
# Used only in encoder Stage 0
# =========================================================
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
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
            nn.Sigmoid(),
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
            ConvBlock(
                channels,
                channels,
                kernel_size=5,
                padding=2,
            ),
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
        )

        self.shortcut_path = nn.Sequential(
            ConvBlock(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBlock(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.se = SEBlock(channels)

    def forward(self, x):
        out = self.deep_path(x) + self.shortcut_path(x) + x
        out = self.se(out)
        return out


# =========================================================
# Single 2D-view bidirectional scan
# Input : [B, C, G, X, Y]
# Output: [B, C, G, X, Y]
# =========================================================
class ToM2DViewScanAware(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.seq_forward = MambaSeq(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.seq_reverse = MambaSeq(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def build_forward_seq(self, x):
        return rearrange(
            x,
            "b c g x y -> b (g x y) c",
        )

    def build_reverse_seq(self, x):
        seq = self.build_forward_seq(x)
        seq = torch.flip(seq, dims=[1])
        return seq

    def seq_to_view(self, seq, G, X, Y):
        return rearrange(
            seq,
            "b (g x y) c -> b c g x y",
            g=G,
            x=X,
            y=Y,
        )

    def forward(self, x):
        B, C, G, X, Y = x.shape

        seq_f = self.build_forward_seq(x)
        out_f = self.seq_forward(seq_f)
        out_f = self.seq_to_view(out_f, G, X, Y)

        seq_r = self.build_reverse_seq(x)
        out_r = self.seq_reverse(seq_r)
        out_r = torch.flip(out_r, dims=[1])
        out_r = self.seq_to_view(out_r, G, X, Y)

        return out_f + out_r


# =========================================================
# Scan-aware gate
# =========================================================
class ScanAwareGate3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.gate = nn.Sequential(
            nn.Conv3d(
                channels * 3,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                hidden_channels,
                channels * 3,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, out_axial, out_coronal, out_sagittal):
        B, C, D, W, H = out_axial.shape

        descriptor = torch.cat(
            [
                out_axial,
                out_coronal,
                out_sagittal,
            ],
            dim=1,
        )

        pooled = self.pool(descriptor)

        gate_logits = self.gate(pooled)
        gate_logits = gate_logits.view(B, 3, C, 1, 1, 1)

        gate_weights = torch.softmax(gate_logits, dim=1)

        w_axial = gate_weights[:, 0]
        w_coronal = gate_weights[:, 1]
        w_sagittal = gate_weights[:, 2]

        out = (
            w_axial * out_axial
            + w_coronal * out_coronal
            + w_sagittal * out_sagittal
        )

        return out


# =========================================================
# ToOM Scan-aware Layer
# 3 orthogonal views x 2 directions = 6 scans
# =========================================================
class ToOMScanAwareLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        gate_reduction=4,
    ):
        super().__init__()

        self.tom_axial = ToM2DViewScanAware(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.tom_coronal = ToM2DViewScanAware(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.tom_sagittal = ToM2DViewScanAware(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.scan_gate = ScanAwareGate3D(
            channels=dim,
            reduction=gate_reduction,
        )

    def forward(self, x):
        # Axial: [B, C, D, W, H]
        out_axial = self.tom_axial(x)

        # Coronal: [B, C, D, W, H] -> [B, C, W, D, H]
        x_coronal = rearrange(
            x,
            "b c d w h -> b c w d h",
        )

        out_coronal = self.tom_coronal(x_coronal)

        out_coronal = rearrange(
            out_coronal,
            "b c w d h -> b c d w h",
        )

        # Sagittal: [B, C, D, W, H] -> [B, C, H, D, W]
        x_sagittal = rearrange(
            x,
            "b c d w h -> b c h d w",
        )

        out_sagittal = self.tom_sagittal(x_sagittal)

        out_sagittal = rearrange(
            out_sagittal,
            "b c h d w -> b c d w h",
        )

        out = self.scan_gate(
            out_axial=out_axial,
            out_coronal=out_coronal,
            out_sagittal=out_sagittal,
        )

        return out


# =========================================================
# Scan-aware Mamba block
# =========================================================
class ScanAwareMambaBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=2.0,
        d_state=16,
        d_conv=4,
        expand=2,
        gate_reduction=4,
    ):
        super().__init__()

        self.norm1 = LayerNormChannelLast3D(dim)

        self.scan = ToOMScanAwareLayer(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            gate_reduction=gate_reduction,
        )

        self.norm2 = LayerNormChannelLast3D(dim)

        self.mlp = MlpChannel(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
        )

    def forward(self, x):
        x = x + self.scan(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ScanAwareMambaStage(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        mlp_ratio=2.0,
        d_state=16,
        d_conv=4,
        expand=2,
        gate_reduction=4,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                ScanAwareMambaBlock3D(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    gate_reduction=gate_reduction,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# =========================================================
# Hybrid Encoder
#
# Stage 0:
#   Downsample + LargeKernelConv
#
# Stage 1/2/3:
#   Downsample + Scan-aware Mamba Stage
# =========================================================
class HybridMambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=48,
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        out_indices=(0, 1, 2, 3),
        mlp_ratio=2.0,
        d_state=16,
        d_conv=4,
        expand=2,
        gate_reduction=4,
    ):
        super().__init__()

        self.out_indices = out_indices

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(
                in_chans,
                dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(dims[0]),
            nn.ReLU(inplace=True),
        )

        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm3d(dims[i + 1]),
                nn.ReLU(inplace=True),
            )

            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        for i in range(4):
            if i == 0:
                stage = nn.Sequential(
                    *[
                        LargeKernelConv(dims[i])
                        for _ in range(depths[i])
                    ]
                )
            else:
                stage = ScanAwareMambaStage(
                    dim=dims[i],
                    depth=depths[i],
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    gate_reduction=gate_reduction,
                )

            self.stages.append(stage)

        self.norms = nn.ModuleList(
            [
                nn.InstanceNorm3d(dims[i])
                for i in range(4)
            ]
        )

        self.mlps = nn.ModuleList(
            [
                MlpChannel(dims[i], 2 * dims[i])
                for i in range(4)
            ]
        )

    def forward_features(self, x):
        outs = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                x_out = self.norms[i](x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


# =========================================================
# Cross-scale alignment
#
# This version keeps channel alignment strategy.
#
# local index:
#   0: enc2, C=96
#   1: enc3, C=192
#   2: enc4, C=384
# =========================================================
class CrossScaleAlign3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        source_idx,
        target_idx,
        norm_name="instance",
    ):
        super().__init__()

        self.source_idx = source_idx
        self.target_idx = target_idx

        if source_idx < target_idx:
            stride = 2 ** (target_idx - source_idx)

            self.op = ConvNormAct3D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            )

            self.need_interp = False

        elif source_idx == target_idx:
            self.op = ConvNormAct3D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            )

            self.need_interp = False

        else:
            self.op = ConvNormAct3D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            )

            self.need_interp = True

    def forward(self, x, target_size):
        x = self.op(x)

        if self.need_interp or x.shape[2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        return x


# =========================================================
# Encoder multi-scale concat aggregation
#
# Channel alignment is retained:
#   each encoder feature -> out_channels
#   concat -> num_levels * out_channels
# =========================================================
class EncoderMultiScaleConcat3D(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        target_idx,
        norm_name="instance",
    ):
        super().__init__()

        self.target_idx = target_idx

        self.align_layers = nn.ModuleList(
            [
                CrossScaleAlign3D(
                    in_channels=encoder_channels[src_idx],
                    out_channels=out_channels,
                    source_idx=src_idx,
                    target_idx=target_idx,
                    norm_name=norm_name,
                )
                for src_idx in range(len(encoder_channels))
            ]
        )

        self.out_channels = out_channels
        self.num_levels = len(encoder_channels)
        self.concat_channels = out_channels * self.num_levels

    def forward(self, encoder_features, target_size):
        aligned_features = []

        for feat, align in zip(encoder_features, self.align_layers):
            aligned = align(feat, target_size)
            aligned_features.append(aligned)

        fused = torch.cat(aligned_features, dim=1)

        return fused


# =========================================================
# Parallel SCSE-style attention refinement
#
# Spatial branch:
#   avg_map = mean(U, dim=channel)
#   max_map = max(U, dim=channel)
#   concat -> Conv7x7x7 -> sigmoid
#
# Channel branch:
#   GAP -> 1x1x1 conv -> ReLU -> 1x1x1 conv -> sigmoid
#
# Fuse:
#   U_scse = U_sse + U_cse
#   reduction: 1x1x1 conv + 3x3x3 conv -> [B, C, D, W, H]
# =========================================================
class ParallelSCSERefine3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        reduction=8,
        norm_name="instance",
        spatial_kernel_size=7,
    ):
        super().__init__()

        hidden = max(in_channels // reduction, 8)

        if spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be odd.")

        spatial_padding = spatial_kernel_size // 2

        # Spatial attention branch:
        # channel mean + channel max -> 7x7x7 conv -> sigmoid
        self.spatial_conv = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=spatial_padding,
            bias=False,
        )

        # Channel attention branch:
        self.channel_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(
                in_channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden,
                in_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.reduction = ChannelReductionBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_name=norm_name,
        )

    def forward(self, U):
        # ---------------------------------------------
        # Spatial attention branch
        # ---------------------------------------------
        avg_map = torch.mean(
            U,
            dim=1,
            keepdim=True,
        )

        max_map, _ = torch.max(
            U,
            dim=1,
            keepdim=True,
        )

        spatial_descriptor = torch.cat(
            [
                avg_map,
                max_map,
            ],
            dim=1,
        )

        spatial_att = torch.sigmoid(
            self.spatial_conv(spatial_descriptor)
        )

        U_sse = U * spatial_att

        # ---------------------------------------------
        # Channel attention branch
        # ---------------------------------------------
        channel_att = self.channel_squeeze(U)

        U_cse = U * channel_att

        # ---------------------------------------------
        # Parallel fusion
        # ---------------------------------------------
        U_scse = U_sse + U_cse

        # Reduce 2C -> C
        out = self.reduction(U_scse)

        return out


# =========================================================
# VCAFv2 Block with pre-reduction before deep path
#
# Original slow version:
#   E_concat: 3C
#   deep_path on 3C channels:
#       5x5x5 -> 3x3x3 -> dilated 3x3x3
#
# New efficient version:
#   E_concat: 3C
#   pre_reduce: 1x1x1, 3C -> 3C / alpha
#   deep_path on 3C / alpha channels
#   final reduction: 3C / alpha -> C
#
# Default alpha = 2.
# =========================================================
class VascularCrossScaleAttentionFusionBlockV2PreReduce3D(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        target_idx,
        norm_name="instance",
        attention_reduction=8,
        spatial_kernel_size=7,
        deep_reduction_alpha=2,
    ):
        super().__init__()

        if deep_reduction_alpha < 1:
            raise ValueError("deep_reduction_alpha must be >= 1.")

        self.encoder_aggregation = EncoderMultiScaleConcat3D(
            encoder_channels=encoder_channels,
            out_channels=out_channels,
            target_idx=target_idx,
            norm_name=norm_name,
        )

        concat_channels = out_channels * len(encoder_channels)

        reduced_channels = max(
            concat_channels // deep_reduction_alpha,
            out_channels,
        )

        self.pre_deep_reduce = ConvNormAct3D(
            in_channels=concat_channels,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            norm_name=norm_name,
            activation=True,
        )

        self.deep_path = nn.Sequential(
            ConvNormAct3D(
                in_channels=reduced_channels,
                out_channels=reduced_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            ),
            ConvNormAct3D(
                in_channels=reduced_channels,
                out_channels=reduced_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            ),
            ConvNormAct3D(
                in_channels=reduced_channels,
                out_channels=reduced_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                norm_name=norm_name,
                activation=True,
            ),
            ChannelReductionBlock3D(
                in_channels=reduced_channels,
                out_channels=out_channels,
                norm_name=norm_name,
            ),
        )

        self.shortcut_path = nn.Sequential(
            ConvNormAct3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            ),
            ConvNormAct3D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                norm_name=norm_name,
                activation=True,
            ),
        )

        self.att_refine = ParallelSCSERefine3D(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            reduction=attention_reduction,
            norm_name=norm_name,
            spatial_kernel_size=spatial_kernel_size,
        )

    def forward(self, dec_up, encoder_features):
        target_size = dec_up.shape[2:]

        E_concat = self.encoder_aggregation(
            encoder_features=encoder_features,
            target_size=target_size,
        )

        E_reduced = self.pre_deep_reduce(E_concat)

        M = self.deep_path(E_reduced)

        N = self.shortcut_path(dec_up)

        U = torch.cat([M, N], dim=1)

        out = self.att_refine(U)

        return out


# =========================================================
# VCAFv2 Up Block with pre-reduction
#
# Used for:
#   dec3: target enc4
#   dec2: target enc3
#   dec1: target enc2
#
# Not used for full-resolution enc1 skip.
# =========================================================
class VCAFv2PreReduceUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        encoder_channels,
        target_idx,
        norm_name="instance",
        attention_reduction=8,
        spatial_kernel_size=7,
        deep_reduction_alpha=2,
    ):
        super().__init__()

        self.target_idx = target_idx

        self.transp_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.fusion = VascularCrossScaleAttentionFusionBlockV2PreReduce3D(
            encoder_channels=encoder_channels,
            out_channels=out_channels,
            target_idx=target_idx,
            norm_name=norm_name,
            attention_reduction=attention_reduction,
            spatial_kernel_size=spatial_kernel_size,
            deep_reduction_alpha=deep_reduction_alpha,
        )

    def forward(self, inp, encoder_features):
        dec_up = self.transp_conv(inp)

        target_feature = encoder_features[self.target_idx]
        target_size = target_feature.shape[2:]

        if dec_up.shape[2:] != target_size:
            dec_up = F.interpolate(
                dec_up,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        out = self.fusion(
            dec_up=dec_up,
            encoder_features=encoder_features,
        )

        return out


# =========================================================
# Final Network
#
# Encoder:
#   Stage 0: LargeKernelConv
#   Stage 1-3: Scan-aware Mamba
#
# Decoder:
#   dec3/dec2/dec1: VCAFv2 SpatialPool + pre-reduced deep path
#   dec0: normal UNETR skip fusion with enc1
#
# VCAFv2 uses only enc2/enc3/enc4 as multi-scale skip pool.
# Full-resolution enc1 is not included in VCAFv2.
# =========================================================
class SegMambaHybridSAVCAFv2SpatialPoolPreReduceNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        feat_size=(48, 96, 192, 384),
        depths=(2, 2, 2, 2),
        hidden_size=None,
        norm_name="instance",
        res_block=True,
        spatial_dims=3,
        mlp_ratio=2.0,
        d_state=16,
        d_conv=4,
        expand=2,
        gate_reduction=4,
        attention_reduction=8,
        spatial_kernel_size=7,
        deep_reduction_alpha=2,
    ) -> None:
        super().__init__()

        if hidden_size is None:
            hidden_size = feat_size[-1] * 2

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feat_size = list(feat_size)
        self.depths = list(depths)
        self.hidden_size = hidden_size

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.input_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.vit = HybridMambaEncoder(
            in_chans=self.feat_size[0],
            depths=self.depths,
            dims=self.feat_size,
            out_indices=(0, 1, 2, 3),
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            gate_reduction=gate_reduction,
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

        # VCAFv2 only uses enc2/enc3/enc4.
        # Local indices:
        #   0: enc2, C=feat_size[1]
        #   1: enc3, C=feat_size[2]
        #   2: enc4, C=feat_size[3]
        vcaf_encoder_channels = [
            self.feat_size[1],
            self.feat_size[2],
            self.feat_size[3],
        ]

        # dec3 target enc4 -> local target_idx=2
        self.decoder5 = VCAFv2PreReduceUpBlock3D(
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            encoder_channels=vcaf_encoder_channels,
            target_idx=2,
            norm_name=norm_name,
            attention_reduction=attention_reduction,
            spatial_kernel_size=spatial_kernel_size,
            deep_reduction_alpha=deep_reduction_alpha,
        )

        # dec2 target enc3 -> local target_idx=1
        self.decoder4 = VCAFv2PreReduceUpBlock3D(
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            encoder_channels=vcaf_encoder_channels,
            target_idx=1,
            norm_name=norm_name,
            attention_reduction=attention_reduction,
            spatial_kernel_size=spatial_kernel_size,
            deep_reduction_alpha=deep_reduction_alpha,
        )

        # dec1 target enc2 -> local target_idx=0
        self.decoder3 = VCAFv2PreReduceUpBlock3D(
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            encoder_channels=vcaf_encoder_channels,
            target_idx=0,
            norm_name=norm_name,
            attention_reduction=attention_reduction,
            spatial_kernel_size=spatial_kernel_size,
            deep_reduction_alpha=deep_reduction_alpha,
        )

        # Full-resolution skip remains standard UNETR fusion.
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

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.num_classes,
        )

    def forward(self, x_in):
        enc1 = self.encoder1(x_in)

        outs = self.vit(enc1)
        x2, x3, x4, x5 = outs

        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)

        enc_hidden = self.encoder5(x5)

        vcaf_encoder_features = [
            enc2,
            enc3,
            enc4,
        ]

        dec3 = self.decoder5(
            enc_hidden,
            vcaf_encoder_features,
        )

        dec2 = self.decoder4(
            dec3,
            vcaf_encoder_features,
        )

        dec1 = self.decoder3(
            dec2,
            vcaf_encoder_features,
        )

        # Normal full-resolution skip fusion.
        dec0 = self.decoder2(
            dec1,
            enc1,
        )

        out = self.decoder1(dec0)

        return self.out(out)


# =========================================================
# Trainer
# =========================================================
class nnUNetTrainerSegmambaHybridSAVCAFv2_SpatialPool_PreReduce(nnUNetTrainerNoDeepSupervision):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )

        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [-1] * len(original_patch_size)

        for i in range(len(original_patch_size)):
            if (original_patch_size[i] / 16) < 1 or ((original_patch_size[i] / 16) % 1) != 0:
                new_patch_size[i] = round(original_patch_size[i] / 16 + 0.5) * 16
            else:
                new_patch_size[i] = original_patch_size[i]

        self.configuration_manager.configuration["patch_size"] = new_patch_size

        self.print_to_log_file(
            f"Patch size changed from {original_patch_size} to {new_patch_size}"
        )

        self.plans_manager.plans["configurations"][self.configuration_name]["patch_size"] = new_patch_size

        self.configuration_manager.configuration["batch_size"] = 1
        self.plans_manager.plans["configurations"][self.configuration_name]["batch_size"] = 1

        self.initial_lr = 1e-4
        self.grad_scaler = None
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)

        num_stages = len(configuration_manager.conv_kernel_sizes)

        features_per_stage = [
            min(
                configuration_manager.UNet_base_num_features * 2 ** i,
                configuration_manager.unet_max_num_features,
            )
            for i in range(num_stages)
        ]

        if len(features_per_stage) < 4:
            raise ValueError(
                f"SegMambaHybridSAVCAFv2_SpatialPool_PreReduce requires at least 4 stages in plans, "
                f"but got {len(features_per_stage)}"
            )

        feat_size = tuple(features_per_stage[:4])

        model = SegMambaHybridSAVCAFv2SpatialPoolPreReduceNet(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            feat_size=feat_size,
            depths=(2, 2, 2, 2),
            hidden_size=feat_size[-1] * 2,
            norm_name="instance",
            res_block=True,
            spatial_dims=len(configuration_manager.patch_size),
            mlp_ratio=2.0,
            d_state=16,
            d_conv=4,
            expand=2,
            gate_reduction=4,
            attention_reduction=8,
            spatial_kernel_size=7,
            deep_reduction_alpha=2,
        )

        return model

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(
            self.device,
            non_blocking=True,
        )

        if isinstance(target, list):
            target = [
                i.to(
                    self.device,
                    non_blocking=True,
                )
                for i in target
            ]
        else:
            target = target.to(
                self.device,
                non_blocking=True,
            )

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)

        loss = self.loss(
            output,
            target,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            12,
        )

        self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy()
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(
            self.device,
            non_blocking=True,
        )

        if isinstance(target, list):
            target = [
                i.to(
                    self.device,
                    non_blocking=True,
                )
                for i in target
            ]
        else:
            target = target.to(
                self.device,
                non_blocking=True,
            )

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)

        del data

        loss = self.loss(
            output,
            target,
        )

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (
                torch.sigmoid(output) > 0.5
            ).long()
        else:
            output_seg = output.argmax(1)[:, None]

            predicted_segmentation_onehot = torch.zeros(
                output.shape,
                device=output.device,
                dtype=torch.float32,
            )

            predicted_segmentation_onehot.scatter_(
                1,
                output_seg,
                1,
            )

            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (
                    target != self.label_manager.ignore_label
                ).float()

                target[
                    target == self.label_manager.ignore_label
                ] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot,
            target,
            axes=axes,
            mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )

        scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            self.num_epochs,
            exponent=1.0,
        )

        self.print_to_log_file(
            f"Using optimizer {optimizer}"
        )

        self.print_to_log_file(
            f"Using scheduler {scheduler}"
        )

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass