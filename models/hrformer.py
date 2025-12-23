"""
HRFormer Backbone Implementation
High-Resolution Transformer for Human Pose Estimation
Reference: https://arxiv.org/abs/2110.09408
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from functools import partial
import math


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows with padding if needed.
    
    Args:
        x: Input tensor (B, H, W, C).
        window_size: Window size.
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # Pad if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    Hp, Wp = H + pad_h, W + pad_w
    
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    return windows, (Hp, Wp)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, Hp: int, Wp: int) -> torch.Tensor:
    """Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size.
        H, W: Original height and width.
        Hp, Wp: Padded height and width.
        
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    
    # Remove padding
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    
    Args:
        dim: Number of input channels.
        window_size: The height and width of the window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value.
        attn_drop: Dropout ratio of attention weight.
        proj_drop: Dropout ratio of output.
        with_rpe: Whether to use relative position encoding.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        with_rpe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.with_rpe = with_rpe
        
        # Relative position bias table
        if with_rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            
            # Get pair-wise relative position index
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (num_windows*B, N, C) where N = window_size * window_size
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class HRFormerBlock(nn.Module):
    """HRFormer Block with Window Attention.
    
    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        with_rpe: Whether to use relative position encoding.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        with_rpe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            with_rpe=with_rpe,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        shortcut = x
        x = self.norm1(x)
        
        # Window partition
        x_windows, (Hp, Wp) = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W, Hp, Wp)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Reshape back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        return x


class Bottleneck(nn.Module):
    """Bottleneck block for Stage 1."""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class HRFormerModule(nn.Module):
    """HRFormer Module with multi-resolution branches."""
    
    def __init__(
        self,
        num_branches: int,
        block: str,
        num_blocks: List[int],
        num_channels: List[int],
        num_heads: List[int],
        mlp_ratios: List[int],
        window_sizes: List[int],
        drop_path_rate: float = 0.,
        with_rpe: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d,
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        # Build branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = self._make_branch(
                branch_index=i,
                block=block,
                num_blocks=num_blocks[i],
                num_channels=num_channels[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                window_size=window_sizes[i],
                drop_path_rate=drop_path_rate,
                with_rpe=with_rpe,
            )
            self.branches.append(branch)
        
        # Build fuse layers
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, norm_layer)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branch(
        self,
        branch_index: int,
        block: str,
        num_blocks: int,
        num_channels: int,
        num_heads: int,
        mlp_ratio: int,
        window_size: int,
        drop_path_rate: float,
        with_rpe: bool,
    ) -> nn.Sequential:
        """Make one branch."""
        layers = []
        
        for i in range(num_blocks):
            if block == 'HRFORMERBLOCK':
                layers.append(
                    HRFormerBlock(
                        dim=num_channels,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=float(mlp_ratio),
                        drop_path=drop_path_rate,
                        with_rpe=with_rpe,
                    )
                )
            else:
                raise ValueError(f"Unknown block type: {block}")
        
        return nn.Sequential(*layers)
    
    def _make_fuse_layers(
        self,
        num_branches: int,
        num_channels: List[int],
        norm_layer: nn.Module,
    ) -> nn.ModuleList:
        """Make fusion layers."""
        if num_branches == 1:
            return None
        
        fuse_layers = nn.ModuleList()
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # Upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        norm_layer(num_channels[i]),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsample
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                norm_layer(num_channels[i]),
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                norm_layer(num_channels[j]),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(fuse_layer)
        
        return fuse_layers
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        # Forward through branches
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        # Fuse
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = None
            for j in range(self.num_branches):
                if j == i:
                    y = x[j] if y is None else y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y_j = self.fuse_layers[i][j](x[j])
                    y_j = F.interpolate(
                        y_j, size=[height_output, width_output],
                        mode='bilinear', align_corners=False
                    )
                    y = y_j if y is None else y + y_j
                else:
                    y_j = self.fuse_layers[i][j](x[j])
                    y = y_j if y is None else y + y_j
            x_fuse.append(self.relu(y))
        
        return x_fuse


class HRFormer(nn.Module):
    """High-Resolution Transformer.
    
    HRFormer backbone for pose estimation.
    
    Args:
        in_channels: Number of input channels.
        extra: Extra configuration dict containing stage settings.
        norm_cfg: Normalization config.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        drop_path_rate: float = 0.2,
        with_rpe: bool = True,
        # Stage 1
        stage1_num_modules: int = 1,
        stage1_num_branches: int = 1,
        stage1_num_blocks: Tuple[int] = (2,),
        stage1_num_channels: Tuple[int] = (64,),
        # Stage 2
        stage2_num_modules: int = 1,
        stage2_num_branches: int = 2,
        stage2_num_blocks: Tuple[int] = (2, 2),
        stage2_num_channels: Tuple[int] = (78, 156),
        stage2_num_heads: Tuple[int] = (2, 4),
        stage2_mlp_ratios: Tuple[int] = (4, 4),
        stage2_window_sizes: Tuple[int] = (7, 7),
        # Stage 3
        stage3_num_modules: int = 4,
        stage3_num_branches: int = 3,
        stage3_num_blocks: Tuple[int] = (2, 2, 2),
        stage3_num_channels: Tuple[int] = (78, 156, 312),
        stage3_num_heads: Tuple[int] = (2, 4, 8),
        stage3_mlp_ratios: Tuple[int] = (4, 4, 4),
        stage3_window_sizes: Tuple[int] = (7, 7, 7),
        # Stage 4
        stage4_num_modules: int = 2,
        stage4_num_branches: int = 4,
        stage4_num_blocks: Tuple[int] = (2, 2, 2, 2),
        stage4_num_channels: Tuple[int] = (78, 156, 312, 624),
        stage4_num_heads: Tuple[int] = (2, 4, 8, 16),
        stage4_mlp_ratios: Tuple[int] = (4, 4, 4, 4),
        stage4_window_sizes: Tuple[int] = (7, 7, 7, 7),
    ):
        super().__init__()
        
        self.drop_path_rate = drop_path_rate
        self.with_rpe = with_rpe
        
        norm_layer = nn.BatchNorm2d
        
        # Stem network
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1 (Bottleneck blocks)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, stage1_num_blocks[0], norm_layer)
        stage1_out_channels = 64 * Bottleneck.expansion  # 256
        
        # Transition 1
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels],
            list(stage2_num_channels),
            norm_layer
        )
        
        # Stage 2
        self.stage2 = self._make_stage(
            num_modules=stage2_num_modules,
            num_branches=stage2_num_branches,
            num_blocks=list(stage2_num_blocks),
            num_channels=list(stage2_num_channels),
            num_heads=list(stage2_num_heads),
            mlp_ratios=list(stage2_mlp_ratios),
            window_sizes=list(stage2_window_sizes),
            norm_layer=norm_layer,
        )
        
        # Transition 2
        self.transition2 = self._make_transition_layer(
            list(stage2_num_channels),
            list(stage3_num_channels),
            norm_layer
        )
        
        # Stage 3
        self.stage3 = self._make_stage(
            num_modules=stage3_num_modules,
            num_branches=stage3_num_branches,
            num_blocks=list(stage3_num_blocks),
            num_channels=list(stage3_num_channels),
            num_heads=list(stage3_num_heads),
            mlp_ratios=list(stage3_mlp_ratios),
            window_sizes=list(stage3_window_sizes),
            norm_layer=norm_layer,
        )
        
        # Transition 3
        self.transition3 = self._make_transition_layer(
            list(stage3_num_channels),
            list(stage4_num_channels),
            norm_layer
        )
        
        # Stage 4
        self.stage4 = self._make_stage(
            num_modules=stage4_num_modules,
            num_branches=stage4_num_branches,
            num_blocks=list(stage4_num_blocks),
            num_channels=list(stage4_num_channels),
            num_heads=list(stage4_num_heads),
            mlp_ratios=list(stage4_mlp_ratios),
            window_sizes=list(stage4_window_sizes),
            norm_layer=norm_layer,
            multi_scale_output=True,
        )
        
        self.out_channels = stage4_num_channels[0]  # 78 for base model
        
        self._init_weights()
    
    def _make_layer(
        self,
        block,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        norm_layer: nn.Module,
    ) -> nn.Sequential:
        """Make bottleneck layer."""
        downsample = None
        if in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, 1, bias=False),
                norm_layer(out_channels * block.expansion),
            )
        
        layers = [block(in_channels, out_channels, downsample=downsample, norm_layer=norm_layer)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(
        self,
        num_channels_pre: List[int],
        num_channels_cur: List[int],
        norm_layer: nn.Module,
    ) -> nn.ModuleList:
        """Make transition layer between stages."""
        num_branches_pre = len(num_channels_pre)
        num_branches_cur = len(num_channels_cur)
        
        transition_layers = nn.ModuleList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        norm_layer(num_channels_cur[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre[-1] if j == 0 else num_channels_cur[i]
                    out_ch = num_channels_cur[i]
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        norm_layer(out_ch),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return transition_layers
    
    def _make_stage(
        self,
        num_modules: int,
        num_branches: int,
        num_blocks: List[int],
        num_channels: List[int],
        num_heads: List[int],
        mlp_ratios: List[int],
        window_sizes: List[int],
        norm_layer: nn.Module,
        multi_scale_output: bool = True,
    ) -> nn.Sequential:
        """Make stage with HRFormer modules."""
        modules = []
        for i in range(num_modules):
            modules.append(
                HRFormerModule(
                    num_branches=num_branches,
                    block='HRFORMERBLOCK',
                    num_blocks=num_blocks,
                    num_channels=num_channels,
                    num_heads=num_heads,
                    mlp_ratios=mlp_ratios,
                    window_sizes=window_sizes,
                    drop_path_rate=self.drop_path_rate,
                    with_rpe=self.with_rpe,
                    norm_layer=norm_layer,
                    multi_scale_output=multi_scale_output or i < num_modules - 1,
                )
            )
        return nn.Sequential(*modules)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Output feature map from highest resolution branch.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition 1 -> Stage 2
        x_list = []
        for i, trans in enumerate(self.transition1):
            if trans is not None:
                x_list.append(trans(x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Transition 2 -> Stage 3
        x_list = []
        for i, trans in enumerate(self.transition2):
            if trans is not None:
                if i < len(y_list):
                    x_list.append(trans(y_list[i]))
                else:
                    x_list.append(trans(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Transition 3 -> Stage 4
        x_list = []
        for i, trans in enumerate(self.transition3):
            if trans is not None:
                if i < len(y_list):
                    x_list.append(trans(y_list[i]))
                else:
                    x_list.append(trans(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Return highest resolution feature map
        return y_list[0]


def hrformer_base(pretrained: bool = False, **kwargs) -> HRFormer:
    """HRFormer-Base model.
    
    Configuration from MMPose hrformer config.
    """
    model = HRFormer(
        in_channels=3,
        drop_path_rate=0.2,
        with_rpe=True,
        # Stage 1
        stage1_num_modules=1,
        stage1_num_branches=1,
        stage1_num_blocks=(2,),
        stage1_num_channels=(64,),
        # Stage 2
        stage2_num_modules=1,
        stage2_num_branches=2,
        stage2_num_blocks=(2, 2),
        stage2_num_channels=(78, 156),
        stage2_num_heads=(2, 4),
        stage2_mlp_ratios=(4, 4),
        stage2_window_sizes=(7, 7),
        # Stage 3
        stage3_num_modules=4,
        stage3_num_branches=3,
        stage3_num_blocks=(2, 2, 2),
        stage3_num_channels=(78, 156, 312),
        stage3_num_heads=(2, 4, 8),
        stage3_mlp_ratios=(4, 4, 4),
        stage3_window_sizes=(7, 7, 7),
        # Stage 4
        stage4_num_modules=2,
        stage4_num_branches=4,
        stage4_num_blocks=(2, 2, 2, 2),
        stage4_num_channels=(78, 156, 312, 624),
        stage4_num_heads=(2, 4, 8, 16),
        stage4_mlp_ratios=(4, 4, 4, 4),
        stage4_window_sizes=(7, 7, 7, 7),
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights
        # checkpoint_url = 'https://download.openmmlab.com/mmpose/pretrain_models/hrformer_base-32815020_20220226.pth'
        pass
    
    return model


def hrformer_small(pretrained: bool = False, **kwargs) -> HRFormer:
    """HRFormer-Small model."""
    model = HRFormer(
        in_channels=3,
        drop_path_rate=0.1,
        with_rpe=True,
        # Stage 2
        stage2_num_channels=(32, 64),
        stage2_num_heads=(1, 2),
        # Stage 3
        stage3_num_channels=(32, 64, 128),
        stage3_num_heads=(1, 2, 4),
        # Stage 4
        stage4_num_channels=(32, 64, 128, 256),
        stage4_num_heads=(1, 2, 4, 8),
        **kwargs
    )
    
    return model
