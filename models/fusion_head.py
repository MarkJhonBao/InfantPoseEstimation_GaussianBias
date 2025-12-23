"""
Fusion Head: Heatmap + Regression with Gaussian Distribution Constraints
用于早产儿姿态估计的融合头部实现

核心特性:
1. Heatmap分支 - 热图预测
2. Regression分支 - 亚像素偏移回归校正
3. Variance分支 - 高斯方差预测
4. 多项损失函数融合
5. 高斯热图分布约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


# ============================================
# Sub-pixel Refinement Module
# ============================================

class SoftArgmax2D(nn.Module):
    """Differentiable soft-argmax for 2D heatmaps.
    
    通过soft-argmax实现可微分的坐标解码，提供亚像素精度。
    
    公式:
        (x̂, ŷ) = Σ (x,y) · exp(H(x,y)) / Σ exp(H(x',y'))
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps: (B, K, H, W)
        Returns:
            coords: (B, K, 2) - (x, y) coordinates
            scores: (B, K) - confidence scores
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Apply beta scaling and softmax
        heatmaps_scaled = heatmaps * self.beta
        heatmaps_flat = heatmaps_scaled.view(B, K, -1)
        heatmaps_softmax = F.softmax(heatmaps_flat, dim=-1).view(B, K, H, W)
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_x = grid_x.view(1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)
        
        # Compute expected coordinates
        x_coords = (heatmaps_softmax * grid_x).sum(dim=[2, 3])
        y_coords = (heatmaps_softmax * grid_y).sum(dim=[2, 3])
        
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        # Get max values as scores
        scores = heatmaps.view(B, K, -1).max(dim=-1)[0]
        
        return coords, scores


class LocalGaussianRefinement(nn.Module):
    """Local Gaussian fitting for sub-pixel refinement.
    
    在预测峰值周围进行局部高斯拟合，提高定位精度。
    """
    
    def __init__(self, local_radius: int = 2):
        super().__init__()
        self.local_radius = local_radius
    
    def forward(
        self,
        heatmaps: torch.Tensor,
        coarse_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            heatmaps: (B, K, H, W)
            coarse_coords: (B, K, 2) - initial coordinate estimates
        Returns:
            refined_coords: (B, K, 2)
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        r = self.local_radius
        
        refined_coords = coarse_coords.clone()
        
        for b in range(B):
            for k in range(K):
                px = int(coarse_coords[b, k, 0].round().clamp(0, W - 1).item())
                py = int(coarse_coords[b, k, 1].round().clamp(0, H - 1).item())
                
                # Extract local patch
                x_min, x_max = max(0, px - r), min(W, px + r + 1)
                y_min, y_max = max(0, py - r), min(H, py + r + 1)
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                local_patch = heatmaps[b, k, y_min:y_max, x_min:x_max]
                
                # Local coordinate grids
                local_y, local_x = torch.meshgrid(
                    torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                    torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                
                # Weighted centroid
                weights = F.softmax(local_patch.flatten(), dim=0).view_as(local_patch)
                refined_coords[b, k, 0] = (weights * local_x).sum()
                refined_coords[b, k, 1] = (weights * local_y).sum()
        
        return refined_coords


class SubPixelRefinement(nn.Module):
    """Combined sub-pixel refinement: Global Soft-Argmax + Local Gaussian + Fusion.
    
    融合全局soft-argmax和局部高斯拟合，实现高精度亚像素定位。
    
    公式:
        x_final = α * x_global + (1-α) * x_local
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        local_radius: int = 2,
        fusion_alpha: float = 0.5,
    ):
        super().__init__()
        self.soft_argmax = SoftArgmax2D(beta=beta)
        self.local_refine = LocalGaussianRefinement(local_radius=local_radius)
        self.alpha = nn.Parameter(torch.tensor(fusion_alpha))
    
    def forward(
        self,
        heatmaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmaps: (B, K, H, W)
        Returns:
            refined_coords: (B, K, 2)
            scores: (B, K)
        """
        # Global soft-argmax
        global_coords, scores = self.soft_argmax(heatmaps)
        
        # Local Gaussian refinement
        local_coords = self.local_refine(heatmaps, global_coords)
        
        # Fusion with learnable alpha
        alpha = torch.sigmoid(self.alpha)
        refined_coords = alpha * global_coords + (1 - alpha) * local_coords
        
        return refined_coords, scores


# ============================================
# Fusion Head: Heatmap + Regression
# ============================================

class HeatmapRegressionHead(nn.Module):
    """Fusion Head: Heatmap + Regression + Variance Prediction.
    
    三分支融合头部：
    1. Heatmap分支：预测关键点热图
    2. Regression分支：预测亚像素偏移量，校正量化误差
       - 量化误差上界: ε_q ≤ (√2/2)s, s为下采样因子
    3. Variance分支：预测高斯分布方差，用于分布约束
    
    Args:
        in_channels: 输入特征通道数
        num_keypoints: 关键点数量
        hidden_dim: 隐藏层维度
        use_subpixel_refinement: 是否使用亚像素精炼
    """
    
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 17,
        hidden_dim: int = 256,
        use_subpixel_refinement: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.use_subpixel_refinement = use_subpixel_refinement
        
        # ============================================
        # Shared Feature Extraction
        # ============================================
        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ============================================
        # Branch 1: Heatmap Prediction
        # ============================================
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_keypoints, 1),
        )
        
        # ============================================
        # Branch 2: Offset Regression (for quantization error correction)
        # 用于校正量化误差的偏移回归分支
        # ============================================
        self.offset_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_keypoints * 2, 1),  # x, y offsets per keypoint
        )
        
        # ============================================
        # Branch 3: Variance Prediction (for Gaussian distribution constraint)
        # 用于高斯分布约束的方差预测分支
        # ============================================
        self.variance_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_keypoints, 1),
            nn.Softplus(),  # Ensure positive variance: σ > 0
        )
        
        # ============================================
        # Sub-pixel Refinement Module
        # ============================================
        if use_subpixel_refinement:
            self.subpixel_refine = SubPixelRefinement(
                beta=1.0,
                local_radius=2,
                fusion_alpha=0.5,
            )
        
        # Learnable fusion weight for heatmap and regression
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Backbone features (B, C, H, W)
            
        Returns:
            Dictionary containing:
                - heatmaps: (B, K, H, W) 关键点热图
                - offsets: (B, K, 2, H, W) 偏移预测
                - variances: (B, K, H, W) 方差预测
                - fusion_weight: scalar 融合权重
        """
        # Shared features
        shared_feat = self.shared_layers(x)
        
        # Branch outputs
        heatmaps = self.heatmap_branch(shared_feat)
        
        offsets = self.offset_branch(shared_feat)
        B, _, H, W = offsets.shape
        offsets = offsets.view(B, self.num_keypoints, 2, H, W)
        
        variances = self.variance_branch(shared_feat)
        
        return {
            'heatmaps': heatmaps,
            'offsets': offsets,
            'variances': variances,
            'fusion_weight': torch.sigmoid(self.fusion_weight),
        }
    
    def decode(
        self,
        outputs: Dict[str, torch.Tensor],
        apply_offset: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode outputs to keypoint coordinates.
        
        解码流程:
        1. Soft-argmax获取初始坐标
        2. 局部高斯精炼
        3. 偏移校正
        
        Args:
            outputs: Head outputs
            apply_offset: Whether to apply offset correction
            
        Returns:
            keypoints: (B, K, 2) in heatmap coordinate space
            scores: (B, K)
        """
        heatmaps = outputs['heatmaps']
        offsets = outputs['offsets']
        
        B, K, H, W = heatmaps.shape
        
        # Sub-pixel coordinate estimation
        if self.use_subpixel_refinement:
            coords, scores = self.subpixel_refine(heatmaps)
        else:
            soft_argmax = SoftArgmax2D()
            coords, scores = soft_argmax(heatmaps)
        
        # Apply offset correction
        if apply_offset:
            # Sample offsets at predicted locations using grid_sample
            coords_normalized = torch.stack([
                2 * coords[:, :, 0] / (W - 1) - 1,
                2 * coords[:, :, 1] / (H - 1) - 1
            ], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
            
            # Reshape for grid_sample
            offsets_reshaped = offsets.view(B * K, 2, H, W)
            coords_reshaped = coords_normalized.view(B * K, 1, 1, 2)
            
            sampled_offsets = F.grid_sample(
                offsets_reshaped,
                coords_reshaped,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).view(B, K, 2)
            
            # Apply weighted offset
            alpha = outputs['fusion_weight']
            coords = coords + alpha * sampled_offsets
        
        return coords, scores


# ============================================
# Gaussian Distribution Constraint Losses
# ============================================

class GaussianDistributionConstraint(nn.Module):
    """Gaussian distribution constraint losses for heatmaps.
    
    高斯分布约束损失，包含：
    1. 方差对齐损失 (Variance Alignment Loss)
       L_variance = Σ |σ_pred - σ_gt|²
    2. 空间重叠正则化 (Spatial Overlap Regularization)
       L_overlap = Σ max(0, overlap_ratio - threshold)
    3. 分布形状约束 (Distribution Shape Constraint)
       通过熵约束鼓励单峰高斯分布
    
    Args:
        target_sigma: 目标高斯sigma值
        overlap_threshold: 重叠阈值
    """
    
    # COCO skeleton connections for overlap computation
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    ]
    
    def __init__(
        self,
        target_sigma: float = 2.0,
        overlap_threshold: float = 0.5,
    ):
        super().__init__()
        self.target_sigma = target_sigma
        self.overlap_threshold = overlap_threshold
    
    def compute_heatmap_variance(
        self,
        heatmaps: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute variance of predicted heatmaps around peak.
        
        计算预测热图相对于峰值位置的方差。
        
        Args:
            heatmaps: (B, K, H, W)
            coords: Peak coordinates (B, K, 2)
            
        Returns:
            sigma: Predicted sigma values (B, K)
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Create coordinate grids
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_x = grid_x.view(1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)
        
        # Normalize heatmaps (make them sum to 1)
        heatmaps_pos = F.relu(heatmaps)  # Ensure non-negative
        heatmaps_norm = heatmaps_pos / (heatmaps_pos.sum(dim=[2, 3], keepdim=True) + 1e-8)
        
        # Peak locations
        mu_x = coords[:, :, 0:1, None]  # (B, K, 1, 1)
        mu_y = coords[:, :, 1:2, None]
        
        # Compute second moment (variance)
        var_x = (heatmaps_norm * (grid_x - mu_x) ** 2).sum(dim=[2, 3])
        var_y = (heatmaps_norm * (grid_y - mu_y) ** 2).sum(dim=[2, 3])
        
        # Combined sigma
        sigma = torch.sqrt(var_x + var_y + 1e-8)
        
        return sigma
    
    def variance_alignment_loss(
        self,
        heatmaps: torch.Tensor,
        coords: torch.Tensor,
        target_weight: torch.Tensor,
        pred_variances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Variance alignment loss.
        
        L_variance = Σ |σ_pred - σ_gt|²
        
        约束预测热图的方差与目标高斯分布一致。
        """
        # Compute variance from heatmaps
        sigma_from_heatmap = self.compute_heatmap_variance(heatmaps, coords)
        
        # Target sigma
        target_sigma = torch.full_like(sigma_from_heatmap, self.target_sigma)
        
        # MSE loss
        loss = (sigma_from_heatmap - target_sigma) ** 2
        
        # If variance prediction branch exists, add its constraint
        if pred_variances is not None:
            # Global average variance prediction
            sigma_pred = pred_variances.mean(dim=[2, 3])
            loss = loss + (sigma_pred - target_sigma) ** 2
        
        # Apply target weight
        weight = target_weight.squeeze(-1)
        loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        
        return loss
    
    def spatial_overlap_loss(
        self,
        heatmaps: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Spatial overlap regularization loss.
        
        L_overlap = Σ max(0, overlap_ratio - threshold)
        
        防止相邻关键点热图过度重叠，减少歧义。
        """
        B, K, H, W = heatmaps.shape
        
        # Normalize heatmaps to probability distributions
        heatmaps_prob = torch.sigmoid(heatmaps)
        
        total_loss = torch.tensor(0.0, device=heatmaps.device)
        count = 0
        
        for (i, j) in self.SKELETON:
            if i >= K or j >= K:
                continue
            
            h_i = heatmaps_prob[:, i]  # (B, H, W)
            h_j = heatmaps_prob[:, j]
            
            # Compute overlap (minimum of two distributions)
            overlap = torch.min(h_i, h_j)
            
            # Overlap ratio relative to smaller heatmap
            h_i_sum = h_i.sum(dim=[1, 2])
            h_j_sum = h_j.sum(dim=[1, 2])
            min_sum = torch.min(h_i_sum, h_j_sum) + 1e-8
            overlap_ratio = overlap.sum(dim=[1, 2]) / min_sum
            
            # Penalize if exceeds threshold
            penalty = F.relu(overlap_ratio - self.overlap_threshold)
            
            # Weight by visibility
            vis_weight = target_weight[:, i, 0] * target_weight[:, j, 0]
            total_loss = total_loss + (penalty * vis_weight).sum()
            count += vis_weight.sum()
        
        return total_loss / (count + 1e-8)
    
    def distribution_shape_loss(
        self,
        heatmaps: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Distribution shape constraint (encourage unimodal Gaussian).
        
        通过熵约束鼓励热图呈现单峰高斯分布。
        低熵意味着更集中的分布（更像高斯）。
        """
        B, K, H, W = heatmaps.shape
        
        # Normalize to probability distribution
        heatmaps_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(heatmaps_flat, dim=-1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Target entropy for Gaussian with target_sigma
        # For a 2D Gaussian: H ≈ log(2πeσ²)
        target_entropy = math.log(2 * math.pi * math.e * self.target_sigma ** 2)
        
        # Penalize deviation from target entropy
        loss = (entropy - target_entropy) ** 2
        
        # Apply weight
        weight = target_weight.squeeze(-1)
        loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        
        return loss
    
    def forward(
        self,
        heatmaps: torch.Tensor,
        coords: torch.Tensor,
        target_weight: torch.Tensor,
        pred_variances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all distribution constraint losses."""
        return {
            'variance_loss': self.variance_alignment_loss(
                heatmaps, coords, target_weight, pred_variances
            ),
            'overlap_loss': self.spatial_overlap_loss(heatmaps, target_weight),
            'shape_loss': self.distribution_shape_loss(heatmaps, target_weight),
        }


# ============================================
# Complete Fusion Loss Function
# ============================================

class FusionPoseLoss(nn.Module):
    """Complete multi-task loss for fusion pose estimation.
    
    总损失函数:
    L_total = λ₁·L_heatmap + λ₂·L_offset + λ₃·L_peak 
            + λ₄·L_variance + λ₅·L_overlap + λ₆·L_shape
    
    其中:
    - L_heatmap: 热图MSE损失
    - L_offset: 偏移回归SmoothL1损失
    - L_peak: 峰值定位欧氏距离损失
    - L_variance: 方差对齐损失
    - L_overlap: 空间重叠正则化损失
    - L_shape: 分布形状约束损失
    
    Args:
        heatmap_weight: 热图损失权重 λ₁
        offset_weight: 偏移回归损失权重 λ₂
        peak_weight: 峰值定位损失权重 λ₃
        variance_weight: 方差对齐损失权重 λ₄
        overlap_weight: 重叠正则化权重 λ₅
        shape_weight: 分布形状约束权重 λ₆
        target_sigma: 目标高斯sigma
        use_target_weight: 是否使用关键点可见性权重
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        peak_weight: float = 0.5,
        variance_weight: float = 0.1,
        overlap_weight: float = 0.05,
        shape_weight: float = 0.05,
        target_sigma: float = 2.0,
        use_target_weight: bool = True,
    ):
        super().__init__()
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.peak_weight = peak_weight
        self.variance_weight = variance_weight
        self.overlap_weight = overlap_weight
        self.shape_weight = shape_weight
        self.use_target_weight = use_target_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.gaussian_constraint = GaussianDistributionConstraint(
            target_sigma=target_sigma
        )
        self.soft_argmax = SoftArgmax2D()
    
    def heatmap_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Heatmap MSE loss.
        
        L_heatmap = (1/K) Σ_k w_k * ||H_pred^k - H_gt^k||²
        """
        B, K, H, W = pred.shape
        
        loss = self.mse_loss(pred, target).mean(dim=[2, 3])  # (B, K)
        
        if self.use_target_weight:
            weight = weight.squeeze(-1)
            loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def offset_loss(
        self,
        pred_offsets: torch.Tensor,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        weight: torch.Tensor,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Offset regression loss.
        
        L_offset = SmoothL1(Δp_pred, Δp_target)
        where Δp_target = p_gt - p_peak
        
        目标偏移 = GT坐标 - 预测峰值坐标
        """
        B, K = pred_coords.shape[:2]
        H, W = heatmap_size
        
        # Scale GT coordinates to heatmap space
        scale_x = heatmap_size[1] / input_size[0]
        scale_y = heatmap_size[0] / input_size[1]
        
        gt_coords_scaled = gt_coords.clone()
        gt_coords_scaled[:, :, 0] = gt_coords[:, :, 0] * scale_x
        gt_coords_scaled[:, :, 1] = gt_coords[:, :, 1] * scale_y
        
        # Target offset = GT - predicted peak
        target_offset = gt_coords_scaled - pred_coords
        
        # Sample predicted offset at peak locations
        coords_norm = torch.stack([
            2 * pred_coords[:, :, 0] / (W - 1) - 1,
            2 * pred_coords[:, :, 1] / (H - 1) - 1
        ], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
        
        pred_offsets_reshaped = pred_offsets.view(B * K, 2, H, W)
        coords_reshaped = coords_norm.view(B * K, 1, 1, 2)
        
        sampled_offset = F.grid_sample(
            pred_offsets_reshaped, coords_reshaped,
            mode='bilinear', padding_mode='border', align_corners=True
        ).view(B, K, 2)
        
        # Compute SmoothL1 loss
        loss = self.smooth_l1(sampled_offset, target_offset).mean(dim=-1)  # (B, K)
        
        if self.use_target_weight:
            weight = weight.squeeze(-1)
            loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def peak_localization_loss(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        weight: torch.Tensor,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Peak localization loss (Euclidean distance).
        
        L_peak = ||p_pred - p_gt||²
        """
        # Scale GT to heatmap space
        scale_x = heatmap_size[1] / input_size[0]
        scale_y = heatmap_size[0] / input_size[1]
        
        gt_scaled = gt_coords.clone()
        gt_scaled[:, :, 0] = gt_coords[:, :, 0] * scale_x
        gt_scaled[:, :, 1] = gt_coords[:, :, 1] * scale_y
        
        # Euclidean distance squared
        loss = ((pred_coords - gt_scaled) ** 2).sum(dim=-1)  # (B, K)
        
        if self.use_target_weight:
            weight = weight.squeeze(-1)
            loss = (loss * weight).sum() / (weight.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_heatmaps: torch.Tensor,
        target_weight: torch.Tensor,
        gt_keypoints: torch.Tensor,
        input_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.
        
        Args:
            outputs: Model head outputs (heatmaps, offsets, variances)
            target_heatmaps: GT heatmaps (B, K, H, W)
            target_weight: Visibility weights (B, K, 1)
            gt_keypoints: GT coordinates (B, K, 2) in input image space
            input_size: Input image size (W, H)
            heatmap_size: Heatmap size (W, H)
            
        Returns:
            Dictionary of all losses including total_loss
        """
        heatmaps = outputs['heatmaps']
        offsets = outputs['offsets']
        variances = outputs['variances']
        
        B, K, H, W = heatmaps.shape
        
        # Get predicted coordinates via soft-argmax
        pred_coords, _ = self.soft_argmax(heatmaps)
        
        losses = {}
        
        # 1. Heatmap MSE loss
        losses['heatmap_loss'] = self.heatmap_weight * self.heatmap_loss(
            heatmaps, target_heatmaps, target_weight
        )
        
        # 2. Offset regression loss
        losses['offset_loss'] = self.offset_weight * self.offset_loss(
            offsets, pred_coords, gt_keypoints, target_weight,
            input_size, (H, W)
        )
        
        # 3. Peak localization loss
        losses['peak_loss'] = self.peak_weight * self.peak_localization_loss(
            pred_coords, gt_keypoints, target_weight,
            input_size, (H, W)
        )
        
        # 4. Gaussian distribution constraints
        gaussian_losses = self.gaussian_constraint(
            heatmaps, pred_coords, target_weight, variances
        )
        losses['variance_loss'] = self.variance_weight * gaussian_losses['variance_loss']
        losses['overlap_loss'] = self.overlap_weight * gaussian_losses['overlap_loss']
        losses['shape_loss'] = self.shape_weight * gaussian_losses['shape_loss']
        
        # Total loss
        losses['total_loss'] = sum(v for k, v in losses.items() if k != 'total_loss')
        
        return losses


# ============================================
# Convenience Builder Functions
# ============================================

def build_fusion_head(
    in_channels: int,
    num_keypoints: int = 17,
    hidden_dim: int = 256,
) -> HeatmapRegressionHead:
    """Build fusion head with default settings."""
    return HeatmapRegressionHead(
        in_channels=in_channels,
        num_keypoints=num_keypoints,
        hidden_dim=hidden_dim,
        use_subpixel_refinement=True,
    )


def build_fusion_loss(
    target_sigma: float = 2.0,
    heatmap_weight: float = 1.0,
    offset_weight: float = 1.0,
    variance_weight: float = 0.1,
) -> FusionPoseLoss:
    """Build fusion loss with default settings."""
    return FusionPoseLoss(
        heatmap_weight=heatmap_weight,
        offset_weight=offset_weight,
        variance_weight=variance_weight,
        target_sigma=target_sigma,
    )
