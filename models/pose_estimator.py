"""
Pose Estimation Head and Complete Model
Supports both standard heatmap head and fusion (heatmap + regression) head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .hrnet import hrnet_w32, hrnet_w48
from .hrformer import hrformer_base, hrformer_small
from .fusion_head import (
    HeatmapRegressionHead,
    FusionPoseLoss,
    SoftArgmax2D,
    build_fusion_head,
    build_fusion_loss,
)


class HeatmapHead(nn.Module):
    """Heatmap prediction head.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (keypoints).
        num_deconv_layers: Number of deconv layers.
        num_deconv_filters: Number of filters in deconv layers.
        num_deconv_kernels: Kernel sizes of deconv layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_deconv_layers: int = 0,
        num_deconv_filters: Tuple[int, ...] = (256, 256, 256),
        num_deconv_kernels: Tuple[int, ...] = (4, 4, 4),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Deconv layers (optional, for SimpleBaseline style)
        if num_deconv_layers > 0:
            layers = []
            for i in range(num_deconv_layers):
                in_ch = in_channels if i == 0 else num_deconv_filters[i - 1]
                out_ch = num_deconv_filters[i]
                kernel = num_deconv_kernels[i]
                padding = (kernel - 1) // 2
                output_padding = kernel - 2 * padding - 2
                
                layers.extend([
                    nn.ConvTranspose2d(
                        in_ch, out_ch, kernel,
                        stride=2, padding=padding,
                        output_padding=output_padding, bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
            
            self.deconv = nn.Sequential(*layers)
            final_in_channels = num_deconv_filters[-1]
        else:
            self.deconv = None
            final_in_channels = in_channels
        
        # Final conv layer
        self.final_layer = nn.Conv2d(
            final_in_channels, out_channels,
            kernel_size=1, stride=1, padding=0
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deconv is not None:
            x = self.deconv(x)
        x = self.final_layer(x)
        return x


class KeypointMSELoss(nn.Module):
    """MSE loss for keypoint heatmaps.
    
    Args:
        use_target_weight: Whether to use target weight.
    """
    
    def __init__(self, use_target_weight: bool = True):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted heatmaps (B, K, H, W).
            target: Target heatmaps (B, K, H, W).
            target_weight: Weight for each keypoint (B, K, 1).
            
        Returns:
            Loss value.
        """
        batch_size = pred.size(0)
        num_keypoints = pred.size(1)
        
        pred = pred.reshape(batch_size, num_keypoints, -1)
        target = target.reshape(batch_size, num_keypoints, -1)
        
        if self.use_target_weight and target_weight is not None:
            loss = self.mse(
                pred * target_weight,
                target * target_weight
            )
        else:
            loss = self.mse(pred, target)
        
        return loss


class PoseEstimator(nn.Module):
    """Top-down Pose Estimator.
    
    Supports two head types:
    1. 'heatmap': Standard heatmap-only head
    2. 'fusion': Heatmap + Regression fusion head with Gaussian constraints
    
    Args:
        backbone: Backbone network name.
        num_keypoints: Number of keypoints.
        pretrained: Whether to use pretrained backbone.
        head_type: 'heatmap' or 'fusion'.
        use_fusion_loss: Whether to use multi-component fusion loss.
    """
    
    def __init__(
        self,
        backbone: str = 'hrformer_base',
        num_keypoints: int = 17,
        pretrained: bool = True,
        head_type: str = 'fusion',
        use_fusion_loss: bool = True,
    ):
        super().__init__()
        
        self.head_type = head_type
        self.use_fusion_loss = use_fusion_loss and (head_type == 'fusion')
        
        # Build backbone
        if backbone == 'hrnet_w32':
            self.backbone = hrnet_w32(pretrained=pretrained)
            in_channels = 32
        elif backbone == 'hrnet_w48':
            self.backbone = hrnet_w48(pretrained=pretrained)
            in_channels = 48
        elif backbone == 'hrformer_base':
            self.backbone = hrformer_base(pretrained=pretrained)
            in_channels = 78  # HRFormer-Base output channels
        elif backbone == 'hrformer_small':
            self.backbone = hrformer_small(pretrained=pretrained)
            in_channels = 32  # HRFormer-Small output channels
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Build head
        if head_type == 'fusion':
            self.head = HeatmapRegressionHead(
                in_channels=in_channels,
                num_keypoints=num_keypoints,
                hidden_dim=256,
                use_subpixel_refinement=True,
            )
            # Fusion loss with multiple components
            self.loss_fn = FusionPoseLoss(
                heatmap_weight=1.0,
                offset_weight=1.0,
                peak_weight=0.5,
                variance_weight=0.1,
                overlap_weight=0.05,
                shape_weight=0.05,
                target_sigma=2.0,
                use_target_weight=True,
            )
        else:
            self.head = HeatmapHead(
                in_channels=in_channels,
                out_channels=num_keypoints,
                num_deconv_layers=0,
            )
            self.loss_fn = KeypointMSELoss(use_target_weight=True)
        
        self.num_keypoints = num_keypoints
        self.soft_argmax = SoftArgmax2D()
    
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_weight: Optional[torch.Tensor] = None,
        gt_keypoints: Optional[torch.Tensor] = None,
        input_size: Tuple[int, int] = (192, 256),
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images (B, 3, H, W).
            target: Target heatmaps (B, K, H', W') for training.
            target_weight: Target weights (B, K, 1) for training.
            gt_keypoints: Ground truth keypoint coordinates (B, K, 2) for fusion loss.
            input_size: Input image size (W, H).
            
        Returns:
            Dictionary containing outputs and optionally losses.
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Head forward
        if self.head_type == 'fusion':
            outputs = self.head(features)
            output = {
                'heatmaps': outputs['heatmaps'],
                'offsets': outputs['offsets'],
                'variances': outputs['variances'],
                'fusion_weight': outputs['fusion_weight'],
            }
        else:
            heatmaps = self.head(features)
            output = {'heatmaps': heatmaps}
        
        # Compute loss if targets provided
        if target is not None:
            if self.head_type == 'fusion' and self.use_fusion_loss and gt_keypoints is not None:
                H, W = output['heatmaps'].shape[2:]
                losses = self.loss_fn(
                    outputs=output,
                    target_heatmaps=target,
                    target_weight=target_weight,
                    gt_keypoints=gt_keypoints,
                    input_size=input_size,
                    heatmap_size=(H, W),
                )
                output['loss'] = losses['total_loss']
                output['losses'] = losses
            else:
                loss = self.loss_fn(output['heatmaps'], target, target_weight)
                output['loss'] = loss
        
        return output
    
    def inference(
        self,
        x: torch.Tensor,
        flip: bool = True,
        flip_pairs: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference with optional flip test.
        
        Args:
            x: Input images (B, 3, H, W).
            flip: Whether to use flip test.
            flip_pairs: Flip pairs for keypoints.
            
        Returns:
            keypoints: Predicted keypoints (B, K, 2) in heatmap space.
            scores: Keypoint scores (B, K).
        """
        # Forward pass
        output = self.forward(x)
        
        if self.head_type == 'fusion':
            # Use fusion head's decode method
            keypoints, scores = self.head.decode(output, apply_offset=True)
            heatmaps = output['heatmaps']
        else:
            heatmaps = output['heatmaps']
            keypoints, scores = self.decode_heatmaps(heatmaps)
        
        if flip and flip_pairs is not None:
            # Flip input and forward
            x_flipped = torch.flip(x, dims=[-1])
            output_flipped = self.forward(x_flipped)
            heatmaps_flipped = output_flipped['heatmaps']
            
            # Flip back heatmaps
            heatmaps_flipped = torch.flip(heatmaps_flipped, dims=[-1])
            
            # Swap left-right keypoints
            heatmaps_flipped_new = heatmaps_flipped.clone()
            for pair in flip_pairs:
                heatmaps_flipped_new[:, pair[0]] = heatmaps_flipped[:, pair[1]]
                heatmaps_flipped_new[:, pair[1]] = heatmaps_flipped[:, pair[0]]
            
            # Average heatmaps
            heatmaps_avg = (heatmaps + heatmaps_flipped_new) / 2
            
            # Decode averaged heatmaps
            if self.head_type == 'fusion':
                output_avg = output.copy()
                output_avg['heatmaps'] = heatmaps_avg
                keypoints, scores = self.head.decode(output_avg, apply_offset=True)
            else:
                keypoints, scores = self.decode_heatmaps(heatmaps_avg)
        
        return keypoints, scores
    
    @staticmethod
    def decode_heatmaps(
        heatmaps: torch.Tensor,
        shift: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode heatmaps to keypoint coordinates (standard method).
        
        Args:
            heatmaps: Predicted heatmaps (B, K, H, W).
            shift: Whether to apply quarter offset shift.
            
        Returns:
            keypoints: Keypoint coordinates (B, K, 2) in heatmap space.
            scores: Keypoint scores (B, K).
        """
        batch_size, num_keypoints, height, width = heatmaps.shape
        
        # Reshape for easier processing
        heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
        
        # Get max indices and values
        max_vals, max_idx = torch.max(heatmaps_flat, dim=2)
        
        # Convert to coordinates
        max_idx = max_idx.float()
        keypoints = torch.zeros(batch_size, num_keypoints, 2, device=heatmaps.device)
        keypoints[:, :, 0] = max_idx % width  # x
        keypoints[:, :, 1] = max_idx // width  # y
        
        # Apply quarter offset shift for sub-pixel accuracy
        if shift:
            for b in range(batch_size):
                for k in range(num_keypoints):
                    x = int(keypoints[b, k, 0].item())
                    y = int(keypoints[b, k, 1].item())
                    
                    if 0 < x < width - 1 and 0 < y < height - 1:
                        diff_x = heatmaps[b, k, y, x + 1] - heatmaps[b, k, y, x - 1]
                        diff_y = heatmaps[b, k, y + 1, x] - heatmaps[b, k, y - 1, x]
                        keypoints[b, k, 0] += torch.sign(diff_x) * 0.25
                        keypoints[b, k, 1] += torch.sign(diff_y) * 0.25
        
        return keypoints, max_vals


def build_model(cfg) -> PoseEstimator:
    """Build pose estimation model.
    
    Args:
        cfg: Configuration object.
        
    Returns:
        PoseEstimator model.
    """
    model = PoseEstimator(
        backbone=cfg.model.backbone,
        num_keypoints=cfg.model.num_keypoints,
        pretrained=cfg.model.pretrained,
        head_type=getattr(cfg.model, 'head_type', 'fusion'),
        use_fusion_loss=getattr(cfg.model, 'use_fusion_loss', True),
    )
    return model
