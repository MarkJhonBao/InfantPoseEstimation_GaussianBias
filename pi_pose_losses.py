"""
Loss Functions for Preterm Infant Pose Estimation
Includes the novel Morphology-Aware Shape Constraint Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedPoseLoss(nn.Module):
    """
    Standard heatmap-based pose estimation loss
    Supports MSE and focal loss variants
    """
    def __init__(self, use_target_weight=True, loss_type='mse'):
        super(FusedPoseLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred_heatmaps, target_heatmaps, target_weight=None):
        """
        Args:
            pred_heatmaps: (B, K, H, W) predicted heatmaps
            target_heatmaps: (B, K, H, W) ground truth heatmaps
            target_weight: (B, K, 1) visibility weights
        """
        batch_size = pred_heatmaps.size(0)
        num_joints = pred_heatmaps.size(1)
        
        # Compute per-pixel loss
        loss = self.criterion(pred_heatmaps, target_heatmaps)
        
        # Apply target weight if provided (for handling occluded/invisible joints)
        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight.view(batch_size, num_joints, 1, 1)
        
        # Average over all dimensions
        loss = loss.mean()
        
        return loss


class MorphologyShapeLoss(nn.Module):
    """
    Morphology-Aware Shape Constraint Loss
    
    Key Innovation: Enforces distribution consistency by penalizing discrepancies
    in spatial variance between predicted and ground-truth keypoint heatmaps.
    
    This addresses:
    - Gaussian bias errors in traditional heatmap representations
    - Peak drift caused by downsampling
    - Unstable limb movement characterization
    
    L_morph = λ * ||Var(P) - Var(GT)||²
    
    Where Var represents the spatial variance (spread) of the heatmap distribution
    """
    def __init__(self, lambda_variance=1.0, lambda_mean=0.5):
        super(MorphologyShapeLoss, self).__init__()
        self.lambda_variance = lambda_variance
        self.lambda_mean = lambda_mean
    
    def compute_spatial_statistics(self, heatmaps):
        """
        Compute spatial mean and variance of heatmap distributions
        
        Args:
            heatmaps: (B, K, H, W) heatmap tensor
        Returns:
            mean: (B, K, 2) spatial center of mass
            variance: (B, K, 2) spatial variance along x and y
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Normalize heatmaps to probability distributions
        heatmaps_flat = heatmaps.view(B, K, -1)
        heatmaps_sum = heatmaps_flat.sum(dim=2, keepdim=True) + 1e-8
        heatmaps_prob = heatmaps_flat / heatmaps_sum
        heatmaps_prob = heatmaps_prob.view(B, K, H, W)
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
        x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)
        
        # Compute spatial means (center of mass)
        mean_y = (heatmaps_prob * y_coords).sum(dim=[2, 3])  # (B, K)
        mean_x = (heatmaps_prob * x_coords).sum(dim=[2, 3])  # (B, K)
        mean = torch.stack([mean_x, mean_y], dim=2)  # (B, K, 2)
        
        # Compute spatial variances
        var_y = (heatmaps_prob * (y_coords - mean_y.view(B, K, 1, 1))**2).sum(dim=[2, 3])
        var_x = (heatmaps_prob * (x_coords - mean_x.view(B, K, 1, 1))**2).sum(dim=[2, 3])
        variance = torch.stack([var_x, var_y], dim=2)  # (B, K, 2)
        
        return mean, variance
    
    def forward(self, pred_heatmaps, target_heatmaps, target_weight=None):
        """
        Args:
            pred_heatmaps: (B, K, H, W) predicted heatmaps
            target_heatmaps: (B, K, H, W) ground truth heatmaps
            target_weight: (B, K, 1) visibility weights
        """
        # Compute spatial statistics
        pred_mean, pred_variance = self.compute_spatial_statistics(pred_heatmaps)
        target_mean, target_variance = self.compute_spatial_statistics(target_heatmaps)
        
        # Variance consistency loss (main component)
        variance_loss = F.mse_loss(pred_variance, target_variance, reduction='none')
        
        # Mean consistency loss (optional, helps with localization)
        mean_loss = F.mse_loss(pred_mean, target_mean, reduction='none')
        
        # Combine losses
        loss = self.lambda_variance * variance_loss + self.lambda_mean * mean_loss
        
        # Apply target weight for visible joints only
        if target_weight is not None:
            batch_size, num_joints = loss.shape[0], loss.shape[1]
            weight = target_weight.view(batch_size, num_joints, 1)
            loss = loss * weight
        
        # Average over batch and joints
        loss = loss.mean()
        
        return loss


class OffsetRegressionLoss(nn.Module):
    """
    Loss for direct coordinate regression branch
    Helps reduce coordinate deviations from downsampling
    """
    def __init__(self, loss_type='smoothl1'):
        super(OffsetRegressionLoss, self).__init__()
        if loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred_coords, target_coords, target_weight=None):
        """
        Args:
            pred_coords: (B, K, 2) predicted coordinates
            target_coords: (B, K, 2) ground truth coordinates
            target_weight: (B, K, 1) visibility weights
        """
        loss = self.criterion(pred_coords, target_coords)
        
        if target_weight is not None:
            batch_size, num_joints = loss.shape[0], loss.shape[1]
            weight = target_weight.view(batch_size, num_joints, 1)
            loss = loss * weight
        
        return loss.mean()


class JointsMSELoss(nn.Module):
    """
    Classic MSE loss for keypoint detection
    Kept for compatibility and baseline comparisons
    """
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class CombinedLoss(nn.Module):
    """
    Combined loss function that integrates all components
    
    Total Loss = w1 * L_heatmap + w2 * L_morph + w3 * L_reg
    
    Where:
    - L_heatmap: Standard heatmap reconstruction loss
    - L_morph: Morphology-aware shape constraint loss (our innovation)
    - L_reg: Direct coordinate regression loss
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        
        self.heatmap_loss = FusedPoseLoss(
            use_target_weight=True,
            loss_type='mse'
        )
        
        self.morph_loss = MorphologyShapeLoss(
            lambda_variance=config.LOSS.MORPH_LAMBDA,
            lambda_mean=0.5
        )
        
        self.regression_loss = OffsetRegressionLoss(loss_type='smoothl1')
        
        # Loss weights
        self.w_heatmap = 1.0
        self.w_morph = config.LOSS.MORPH_WEIGHT
        self.w_reg = config.LOSS.REG_WEIGHT
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'heatmaps', 'coords', 'refined_coords'
            targets: dict with 'heatmaps', 'coords', 'weights'
        """
        losses = {}
        
        # Heatmap loss
        if 'heatmaps' in predictions and 'heatmaps' in targets:
            losses['heatmap'] = self.heatmap_loss(
                predictions['heatmaps'],
                targets['heatmaps'],
                targets.get('weights')
            )
        
        # Morphology loss (key innovation)
        if 'heatmaps' in predictions and 'heatmaps' in targets:
            losses['morph'] = self.morph_loss(
                predictions['heatmaps'],
                targets['heatmaps'],
                targets.get('weights')
            )
        
        # Regression loss
        if 'coords' in predictions and 'coords' in targets:
            losses['regression'] = self.regression_loss(
                predictions['coords'],
                targets['coords'],
                targets.get('weights')
            )
        
        # Refined coordinates loss (if available)
        if 'refined_coords' in predictions and 'coords' in targets:
            losses['refined'] = self.regression_loss(
                predictions['refined_coords'],
                targets['coords'],
                targets.get('weights')
            )
        
        # Total loss
        total_loss = (
            self.w_heatmap * losses.get('heatmap', 0) +
            self.w_morph * losses.get('morph', 0) +
            self.w_reg * losses.get('regression', 0) +
            self.w_reg * losses.get('refined', 0)
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


# Factory function
def build_loss(config):
    """Build loss function based on configuration"""
    return CombinedLoss(config)
