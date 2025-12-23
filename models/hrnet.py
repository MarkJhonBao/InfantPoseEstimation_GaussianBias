"""
HRNet Backbone Implementation
High-Resolution Network for Human Pose Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BasicBlock(nn.Module):
    """Basic residual block."""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block."""
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
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


class HighResolutionModule(nn.Module):
    """High-Resolution Module with multi-scale fusion."""
    
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_channels: List[int],
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_one_branch(
        self,
        branch_index: int,
        block: nn.Module,
        num_blocks: int,
        num_channels: int,
    ) -> nn.Sequential:
        """Make one branch."""
        layers = []
        for i in range(num_blocks):
            layers.append(
                block(num_channels, num_channels)
            )
        return nn.Sequential(*layers)
    
    def _make_branches(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_channels: List[int],
    ) -> nn.ModuleList:
        """Make all branches."""
        branches = nn.ModuleList()
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks[i], num_channels[i])
            )
        return branches
    
    def _make_fuse_layers(
        self,
        num_branches: int,
        num_channels: List[int],
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
                        nn.BatchNorm2d(num_channels[i]),
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
                                nn.BatchNorm2d(num_channels[i]),
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
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


class HRNet(nn.Module):
    """High-Resolution Network.
    
    Args:
        in_channels: Number of input channels.
        base_channels: Base number of channels.
        num_stages: Number of stages.
        num_modules: Number of modules per stage.
        num_branches: Number of branches per stage.
        num_blocks: Number of blocks per branch.
        num_channels: Number of channels per branch.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
        
        # Transition 1
        self.transition1 = self._make_transition_layer(
            [256], [base_channels, base_channels * 2]
        )
        
        # Stage 2
        self.stage2 = self._make_stage(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=[base_channels, base_channels * 2],
        )
        
        # Transition 2
        self.transition2 = self._make_transition_layer(
            [base_channels, base_channels * 2],
            [base_channels, base_channels * 2, base_channels * 4]
        )
        
        # Stage 3
        self.stage3 = self._make_stage(
            num_modules=4,
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=[base_channels, base_channels * 2, base_channels * 4],
        )
        
        # Transition 3
        self.transition3 = self._make_transition_layer(
            [base_channels, base_channels * 2, base_channels * 4],
            [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        )
        
        # Stage 4
        self.stage4 = self._make_stage(
            num_modules=3,
            num_branches=4,
            num_blocks=[4, 4, 4, 4],
            num_channels=[base_channels, base_channels * 2, base_channels * 4, base_channels * 8],
            multi_scale_output=True,
        )
        
        self._init_weights()
    
    def _make_layer(
        self,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
    ) -> nn.Sequential:
        """Make a layer of blocks."""
        downsample = None
        if in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * block.expansion,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = [block(in_channels, out_channels, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(
        self,
        num_channels_pre: List[int],
        num_channels_cur: List[int],
    ) -> nn.ModuleList:
        """Make transition layer."""
        num_branches_pre = len(num_channels_pre)
        num_branches_cur = len(num_channels_cur)
        
        transition_layers = nn.ModuleList()
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
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
                        nn.BatchNorm2d(out_ch),
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
        multi_scale_output: bool = True,
    ) -> nn.Sequential:
        """Make a stage."""
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=BasicBlock,
                    num_blocks=num_blocks,
                    num_channels=num_channels,
                    multi_scale_output=multi_scale_output or i < num_modules - 1,
                )
            )
        return nn.Sequential(*modules)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition 1 & Stage 2
        x_list = []
        for i, transition in enumerate(self.transition1):
            if transition is not None:
                x_list.append(transition(x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Transition 2 & Stage 3
        x_list = []
        for i, transition in enumerate(self.transition2):
            if transition is not None:
                if i < len(y_list):
                    x_list.append(transition(y_list[i]))
                else:
                    x_list.append(transition(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Transition 3 & Stage 4
        x_list = []
        for i, transition in enumerate(self.transition3):
            if transition is not None:
                if i < len(y_list):
                    x_list.append(transition(y_list[i]))
                else:
                    x_list.append(transition(y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Return highest resolution feature
        return y_list[0]


def hrnet_w32(pretrained: bool = False) -> HRNet:
    """HRNet-W32."""
    model = HRNet(base_channels=32)
    if pretrained:
        # Load pretrained weights if available
        pass
    return model


def hrnet_w48(pretrained: bool = False) -> HRNet:
    """HRNet-W48."""
    model = HRNet(base_channels=48)
    if pretrained:
        pass
    return model
