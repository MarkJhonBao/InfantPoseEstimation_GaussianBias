"""
Post-processing utilities for preterm infant pose estimation
Implements the fused decoding strategy combining heatmap and regression
"""
import numpy as np
import torch
import torch.nn.functional as F


def get_max_preds(batch_heatmaps):
    """
    Get predictions from heatmaps using argmax
    
    Args:
        batch_heatmaps: (B, K, H, W) heatmap tensor
    Returns:
        preds: (B, K, 2) predicted coordinates
        maxvals: (B, K, 1) confidence scores
    """
    batch_size, num_joints, height, width = batch_heatmaps.shape
    
    # Reshape to (B, K, H*W)
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    
    # Get max value and index
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    maxvals = maxvals.unsqueeze(dim=-1)  # (B, K, 1)
    
    # Convert flat index to 2D coordinates
    preds = torch.zeros((batch_size, num_joints, 2), dtype=torch.float32, device=batch_heatmaps.device)
    preds[:, :, 0] = idx % width  # x coordinate
    preds[:, :, 1] = idx // width  # y coordinate
    
    return preds, maxvals


def get_max_preds_with_subpixel(batch_heatmaps):
    """
    Get predictions from heatmaps with sub-pixel refinement
    Uses Taylor expansion to refine the peak location
    
    This addresses coordinate deviations from feature map downsampling
    """
    batch_size, num_joints, height, width = batch_heatmaps.shape
    
    # Get initial max predictions
    preds, maxvals = get_max_preds(batch_heatmaps)
    
    # Refine with sub-pixel accuracy
    for b in range(batch_size):
        for k in range(num_joints):
            hm = batch_heatmaps[b, k]
            px = int(preds[b, k, 0])
            py = int(preds[b, k, 1])
            
            # Check boundaries
            if 1 < px < width - 1 and 1 < py < height - 1:
                # Calculate derivatives using neighboring pixels
                diff_x = (hm[py, px + 1] - hm[py, px - 1]).item()
                diff_y = (hm[py + 1, px] - hm[py - 1, px]).item()
                
                # Second derivatives
                diff_xx = (hm[py, px + 1] - 2 * hm[py, px] + hm[py, px - 1]).item()
                diff_yy = (hm[py + 1, px] - 2 * hm[py, px] + hm[py - 1, px]).item()
                
                # Sub-pixel offsets (avoid division by zero)
                if diff_xx < 0:
                    offset_x = diff_x / (2 * abs(diff_xx))
                    preds[b, k, 0] += np.clip(offset_x, -0.5, 0.5)
                
                if diff_yy < 0:
                    offset_y = diff_y / (2 * abs(diff_yy))
                    preds[b, k, 1] += np.clip(offset_y, -0.5, 0.5)
    
    return preds, maxvals


def fused_decode(heatmaps, regression_coords=None, centers=None, scales=None, alpha=0.5):
    """
    Fused decoding strategy - Key Innovation
    
    Combines heatmap-based and regression-based representations to enhance
    scale robustness against diverse body sizes and subtle motions.
    
    Final_Coords = α × Heatmap_Coords + (1-α) × Regression_Coords
    
    Args:
        heatmaps: (B, K, H, W) predicted heatmaps
        regression_coords: (B, K, 2) optional direct regression coordinates
        centers: (B, 2) bounding box centers
        scales: (B, 2) bounding box scales
        alpha: fusion weight for heatmap predictions (0.5 = equal weight)
    
    Returns:
        preds: (B, K, 2) final predicted coordinates in original image space
        maxvals: (B, K, 1) confidence scores
    """
    # Get heatmap predictions with sub-pixel refinement
    heatmap_preds, maxvals = get_max_preds_with_subpixel(heatmaps)
    
    batch_size, num_joints, _ = heatmap_preds.shape
    _, _, height, width = heatmaps.shape
    
    # Scale heatmap coordinates to image space
    if centers is not None and scales is not None:
        # Transform from heatmap space to original image space
        for b in range(batch_size):
            # Scale to image resolution (assuming 256x256 input)
            image_size = 256  # Should match config
            scale_x = image_size / width
            scale_y = image_size / height
            
            heatmap_preds[b, :, 0] *= scale_x
            heatmap_preds[b, :, 1] *= scale_y
    
    # Fuse with regression coordinates if available
    if regression_coords is not None:
        # Ensure both are in same coordinate space
        # Regression coords are typically normalized [0, 1]
        if regression_coords.max() <= 1.0:
            # Scale regression coords to image space
            image_size = 256
            regression_coords = regression_coords * image_size
        
        # Weighted fusion
        preds = alpha * heatmap_preds + (1 - alpha) * regression_coords
        
        # Confidence-based adaptive fusion (optional enhancement)
        # Use higher weight for the prediction with higher confidence
        adaptive_alpha = maxvals / (maxvals + 0.1)  # Normalize confidence
        preds = adaptive_alpha * heatmap_preds + (1 - adaptive_alpha) * regression_coords
    else:
        preds = heatmap_preds
    
    return preds, maxvals


def coordinate_refinement(heatmaps, initial_coords, window_size=5):
    """
    Additional coordinate refinement to reduce localization errors
    
    Uses local window around predicted location to compute weighted average
    This helps under conditions of poor feature diversity
    """
    batch_size, num_joints, height, width = heatmaps.shape
    refined_coords = initial_coords.clone()
    
    half_window = window_size // 2
    
    for b in range(batch_size):
        for k in range(num_joints):
            x = int(initial_coords[b, k, 0].item())
            y = int(initial_coords[b, k, 1].item())
            
            # Define window bounds
            x_min = max(0, x - half_window)
            x_max = min(width, x + half_window + 1)
            y_min = max(0, y - half_window)
            y_max = min(height, y + half_window + 1)
            
            # Extract local window
            local_heatmap = heatmaps[b, k, y_min:y_max, x_min:x_max]
            
            if local_heatmap.numel() == 0:
                continue
            
            # Compute weighted average position
            h_local, w_local = local_heatmap.shape
            
            # Create coordinate grids
            y_coords = torch.arange(y_min, y_max, dtype=torch.float32, device=heatmaps.device)
            x_coords = torch.arange(x_min, x_max, dtype=torch.float32, device=heatmaps.device)
            
            # Normalize heatmap to weights
            weights = local_heatmap / (local_heatmap.sum() + 1e-8)
            
            # Weighted average
            refined_x = (weights.sum(dim=0) * x_coords).sum()
            refined_y = (weights.sum(dim=1) * y_coords).sum()
            
            refined_coords[b, k, 0] = refined_x
            refined_coords[b, k, 1] = refined_y
    
    return refined_coords


def temporal_smoothing(coords_sequence, window_size=5, method='gaussian'):
    """
    Temporal smoothing for video sequences
    
    Addresses unstable limb movements and noise interference mentioned in paper
    Applies smoothing across frames to stabilize movement trajectories
    
    Args:
        coords_sequence: (T, K, 2) sequence of joint coordinates over time
        window_size: smoothing window size
        method: 'gaussian' or 'moving_average'
    """
    T, K, _ = coords_sequence.shape
    smoothed = coords_sequence.clone()
    
    if method == 'gaussian':
        # Gaussian kernel weights
        sigma = window_size / 3.0
        kernel = np.exp(-np.arange(window_size)**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
    else:
        # Uniform weights
        kernel = np.ones(window_size) / window_size
    
    half_window = window_size // 2
    
    for k in range(K):
        for dim in range(2):  # x and y
            trajectory = coords_sequence[:, k, dim].cpu().numpy()
            
            # Apply convolution with padding
            padded = np.pad(trajectory, (half_window, half_window), mode='edge')
            smoothed_traj = np.convolve(padded, kernel, mode='valid')
            
            smoothed[:, k, dim] = torch.from_numpy(smoothed_traj).to(coords_sequence.device)
    
    return smoothed


def filter_low_confidence(preds, maxvals, threshold=0.3):
    """
    Filter out low-confidence predictions
    
    Args:
        preds: (B, K, 2) predicted coordinates
        maxvals: (B, K, 1) confidence scores
        threshold: minimum confidence threshold
    """
    mask = (maxvals > threshold).float()
    filtered_preds = preds * mask
    
    return filtered_preds, mask


def nms_pose(preds, maxvals, distance_threshold=5.0):
    """
    Non-maximum suppression for pose predictions
    Useful when multiple detections are present
    """
    batch_size, num_joints, _ = preds.shape
    keep_mask = torch.ones((batch_size, num_joints, 1), dtype=torch.bool, device=preds.device)
    
    for b in range(batch_size):
        for k in range(num_joints):
            if not keep_mask[b, k, 0]:
                continue
            
            # Find nearby predictions
            distances = torch.sqrt(((preds[b] - preds[b, k])**2).sum(dim=1))
            nearby = distances < distance_threshold
            
            # Keep only the one with highest confidence
            nearby_confs = maxvals[b, nearby, 0]
            if len(nearby_confs) > 1:
                max_idx = torch.argmax(nearby_confs)
                nearby_indices = torch.where(nearby)[0]
                for i, idx in enumerate(nearby_indices):
                    if i != max_idx:
                        keep_mask[b, idx, 0] = False
    
    return preds * keep_mask.float(), keep_mask


def transform_preds(coords, center, scale, output_size, input_size=[256, 256]):
    """
    Transform predictions from model output space to original image space
    
    Args:
        coords: (B, K, 2) coordinates in model space
        center: (B, 2) crop center
        scale: (B, 2) crop scale
        output_size: original image size
        input_size: model input size
    """
    batch_size = coords.shape[0]
    target_coords = coords.clone()
    
    for b in range(batch_size):
        # Scale from model space to crop space
        scale_x = scale[b, 0] / input_size[0]
        scale_y = scale[b, 1] / input_size[1]
        
        target_coords[b, :, 0] = coords[b, :, 0] * scale_x + center[b, 0] - scale[b, 0] / 2
        target_coords[b, :, 1] = coords[b, :, 1] * scale_y + center[b, 1] - scale[b, 1] / 2
    
    return target_coords


# Factory function for complete post-processing pipeline
def postprocess_predictions(outputs, batch_meta, config):
    """
    Complete post-processing pipeline
    
    Args:
        outputs: dict with 'heatmaps', 'coords', 'refined_coords'
        batch_meta: dict with 'center', 'scale', 'image_id'
        config: configuration object
    """
    heatmaps = outputs['heatmaps']
    regression_coords = outputs.get('coords', None)
    
    # Fused decoding (key innovation)
    preds, maxvals = fused_decode(
        heatmaps,
        regression_coords,
        batch_meta.get('center'),
        batch_meta.get('scale'),
        alpha=config.TEST.FUSION_ALPHA if hasattr(config.TEST, 'FUSION_ALPHA') else 0.5
    )
    
    # Additional refinement
    preds = coordinate_refinement(heatmaps, preds)
    
    # Filter low confidence
    preds, mask = filter_low_confidence(preds, maxvals, threshold=0.3)
    
    # Transform to original image space
    if 'center' in batch_meta and 'scale' in batch_meta:
        preds = transform_preds(
            preds,
            batch_meta['center'],
            batch_meta['scale'],
            output_size=[640, 480]  # Original image size
        )
    
    return {
        'preds': preds,
        'maxvals': maxvals,
        'mask': mask
    }
