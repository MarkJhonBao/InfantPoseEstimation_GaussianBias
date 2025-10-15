"""
HRNet-based Pose Estimation Model with Fused Head
Combines heatmap-based and regression-based predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_branches(self, num_branches, block, num_blocks, num_inchannels, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_inchannels, num_channels))
        return nn.ModuleList(branches)

    def _make_one_branch(self, branch_index, block, num_blocks, num_inchannels, num_channels):
        downsample = None
        if num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(num_inchannels[branch_index],
                            num_channels[branch_index], downsample=downsample))
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index] * block.expansion,
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i],
                                  kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i],
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(self.num_inchannels[i])))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.num_inchannels[j], self.num_inchannels[j],
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(self.num_inchannels[j]),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:], mode='bilinear', align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class FusedHeadModule(nn.Module):
    """
    Fused Head that outputs both heatmaps and direct coordinates
    Key innovation: combines benefits of both representations
    """
    def __init__(self, in_channels, num_joints, heatmap_size):
        super(FusedHeadModule, self).__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size
        
        # Heatmap branch (traditional)
        self.heatmap_head = nn.Conv2d(
            in_channels, num_joints,
            kernel_size=1, stride=1, padding=0
        )
        
        # Regression branch (direct coordinate prediction)
        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 2, num_joints * 2)  # (x, y) for each joint
        )
        
        # Coordinate refinement module
        self.refinement = CoordinateRefinementModule(in_channels, num_joints)
        
    def forward(self, x):
        # Generate heatmaps
        heatmaps = self.heatmap_head(x)
        
        # Direct coordinate regression
        coords = self.regression_head(x)
        coords = coords.view(-1, self.num_joints, 2)
        
        # Refine coordinates using both representations
        refined_coords = self.refinement(x, heatmaps, coords)
        
        return {
            'heatmaps': heatmaps,
            'coords': coords,
            'refined_coords': refined_coords
        }


class CoordinateRefinementModule(nn.Module):
    """
    Refinement mechanism that jointly optimizes heatmap and regression outputs
    Reduces localization errors under poor feature diversity
    """
    def __init__(self, in_channels, num_joints):
        super(CoordinateRefinementModule, self).__init__()
        self.num_joints = num_joints
        
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels + num_joints, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_joints * 2, kernel_size=1)
        )
        
    def forward(self, features, heatmaps, reg_coords):
        # Concatenate features with heatmaps
        combined = torch.cat([features, heatmaps], dim=1)
        
        # Predict offset corrections
        offsets = self.offset_predictor(combined)
        B, _, H, W = offsets.shape
        offsets = offsets.view(B, self.num_joints, 2, H, W)
        
        # Sample offsets at regression coordinate locations
        # This creates correspondence between the two representations
        offsets = offsets.mean(dim=[3, 4])  # Simplified - can use spatial sampling
        
        # Apply offsets to regression coordinates
        refined = reg_coords + offsets * 0.1  # Small correction factor
        
        return refined


class PoseHighResolutionNet(nn.Module):
    """
    Main HRNet model with fused head for preterm infant pose estimation
    """
    def __init__(self, config):
        super(PoseHighResolutionNet, self).__init__()
        self.config = config
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        
        # Build HR stages (simplified version)
        # In full implementation, add multiple HR stages
        self.stage2 = self._make_stage(2, [32, 64], [4, 4])
        self.stage3 = self._make_stage(3, [32, 64, 128], [4, 4, 4])
        self.stage4 = self._make_stage(4, [32, 64, 128, 256], [4, 4, 4, 4])
        
        # Fused head
        if config.MODEL.FUSED_HEAD:
            self.final_layer = FusedHeadModule(
                in_channels=32,  # Highest resolution branch
                num_joints=config.MODEL.NUM_JOINTS,
                heatmap_size=config.MODEL.HEATMAP_SIZE
            )
        else:
            # Traditional heatmap-only head
            self.final_layer = nn.Conv2d(
                32, config.MODEL.NUM_JOINTS,
                kernel_size=1, stride=1, padding=0
            )
        
        self.init_weights()
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        return nn.Sequential(*layers)
    
    def _make_stage(self, num_branches, num_channels, num_blocks):
        # Simplified stage creation
        # Full implementation would include proper transition layers
        return HighResolutionModule(
            num_branches=num_branches,
            blocks=BasicBlock,
            num_blocks=num_blocks,
            num_inchannels=num_channels,
            num_channels=num_channels,
            fuse_method='SUM',
            multi_scale_output=True
        )
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # HR Stages (simplified)
        x = [x]  # Start with single resolution
        # In full implementation, properly handle multi-resolution
        
        # Use highest resolution feature for final prediction
        x = x[0]
        
        # Final layer
        output = self.final_layer(x)
        
        # Return dict or tensor based on head type
        if isinstance(output, dict):
            return output
        else:
            return {'heatmaps': output}


def get_pose_net(config, is_train=True):
    """Factory function to create model"""
    model = PoseHighResolutionNet(config)
    
    if is_train and config.MODEL.PRETRAINED:
        # Load pretrained weights if available
        pass
    
    return model
