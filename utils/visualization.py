"""
Visualization Utilities for Pose Estimation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# COCO keypoint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

# Colors for keypoints (BGR)
COCO_COLORS = [
    (255, 0, 0),      # nose - blue
    (255, 85, 0),     # left_eye
    (255, 170, 0),    # right_eye
    (255, 255, 0),    # left_ear
    (170, 255, 0),    # right_ear
    (85, 255, 0),     # left_shoulder
    (0, 255, 0),      # right_shoulder - green
    (0, 255, 85),     # left_elbow
    (0, 255, 170),    # right_elbow
    (0, 255, 255),    # left_wrist
    (0, 170, 255),    # right_wrist
    (0, 85, 255),     # left_hip
    (0, 0, 255),      # right_hip - red
    (85, 0, 255),     # left_knee
    (170, 0, 255),    # right_knee
    (255, 0, 255),    # left_ankle
    (255, 0, 170),    # right_ankle
]


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    score_threshold: float = 0.3,
    skeleton: List[Tuple[int, int]] = COCO_SKELETON,
    colors: List[Tuple[int, int, int]] = COCO_COLORS,
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw skeleton on image.
    
    Args:
        img: Input image (H, W, 3) in BGR format.
        keypoints: Keypoint coordinates (K, 2).
        scores: Keypoint scores (K,).
        score_threshold: Minimum score to draw keypoint.
        skeleton: Skeleton connections.
        colors: Colors for each keypoint.
        point_radius: Radius of keypoint circle.
        line_thickness: Thickness of skeleton lines.
        
    Returns:
        Image with skeleton drawn.
    """
    img = img.copy()
    
    if scores is None:
        scores = np.ones(len(keypoints))
    
    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if scores[start_idx] >= score_threshold and scores[end_idx] >= score_threshold:
            start_pt = tuple(keypoints[start_idx].astype(int))
            end_pt = tuple(keypoints[end_idx].astype(int))
            
            # Use color of the starting keypoint
            color = colors[start_idx % len(colors)]
            
            cv2.line(img, start_pt, end_pt, color, line_thickness)
    
    # Draw keypoints
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        if score >= score_threshold:
            x, y = int(kpt[0]), int(kpt[1])
            color = colors[i % len(colors)]
            cv2.circle(img, (x, y), point_radius, color, -1)
            cv2.circle(img, (x, y), point_radius + 1, (255, 255, 255), 1)
    
    return img


def draw_heatmaps(
    img: np.ndarray,
    heatmaps: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay heatmaps on image.
    
    Args:
        img: Input image (H, W, 3) in BGR format.
        heatmaps: Heatmaps (K, H', W').
        alpha: Blending alpha.
        
    Returns:
        Image with heatmaps overlaid.
    """
    img = img.copy()
    h, w = img.shape[:2]
    
    # Sum all heatmaps
    heatmap_sum = heatmaps.max(axis=0)
    
    # Resize to image size
    heatmap_sum = cv2.resize(heatmap_sum, (w, h))
    
    # Normalize to 0-255
    heatmap_sum = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-8)
    heatmap_sum = (heatmap_sum * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_sum, cv2.COLORMAP_JET)
    
    # Blend with original image
    result = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    
    return result


def draw_bbox(
    img: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box on image.
    
    Args:
        img: Input image.
        bbox: Bounding box (x1, y1, x2, y2).
        color: Box color.
        thickness: Line thickness.
        
    Returns:
        Image with bbox drawn.
    """
    img = img.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def create_grid_image(
    images: List[np.ndarray],
    ncols: int = 4,
    padding: int = 2,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create grid image from list of images.
    
    Args:
        images: List of images.
        ncols: Number of columns.
        padding: Padding between images.
        bg_color: Background color.
        
    Returns:
        Grid image.
    """
    if len(images) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Ensure all images have same size
    h, w = images[0].shape[:2]
    
    nrows = (len(images) + ncols - 1) // ncols
    
    grid_h = nrows * h + (nrows + 1) * padding
    grid_w = ncols * w + (ncols + 1) * padding
    
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        
        # Resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        grid[y:y+h, x:x+w] = img
    
    return grid


def save_visualization(
    img: np.ndarray,
    output_path: str,
    keypoints: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    heatmaps: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
):
    """Save visualization to file.
    
    Args:
        img: Input image.
        output_path: Output file path.
        keypoints: Optional keypoints to draw.
        scores: Optional keypoint scores.
        heatmaps: Optional heatmaps to overlay.
        bbox: Optional bounding box.
    """
    result = img.copy()
    
    if bbox is not None:
        result = draw_bbox(result, bbox)
    
    if heatmaps is not None:
        result = draw_heatmaps(result, heatmaps, alpha=0.3)
    
    if keypoints is not None:
        result = draw_skeleton(result, keypoints, scores)
    
    cv2.imwrite(output_path, result)
