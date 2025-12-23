"""
Human Pose Estimation Project
Based on MMPose configuration - Pure PyTorch Implementation
"""

# ============================================
# configs/config.py
# ============================================

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_root: str = 'data/coco/'
    train_ann: str = 'annotations/person_keypoints_train2017.json'
    val_ann: str = 'annotations/person_keypoints_val2017.json'
    train_img_prefix: str = 'train2017/'
    val_img_prefix: str = 'val2017/'
    
    # Input/Output sizes
    input_size: Tuple[int, int] = (192, 256)  # (width, height)
    heatmap_size: Tuple[int, int] = (48, 64)  # (width, height)
    
    # Keypoint settings
    num_keypoints: int = 17
    sigma: float = 2.0
    
    # COCO keypoint names
    keypoint_names: List[str] = field(default_factory=lambda: [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ])
    
    # Flip pairs for augmentation
    flip_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
    ])


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = 'hrformer_base'
    pretrained: bool = True
    
    # HRFormer settings
    in_channels: int = 3
    base_channels: int = 78  # HRFormer-Base
    
    # Head settings
    head_type: str = 'fusion'  # 'heatmap' or 'fusion'
    head_in_channels: int = 78  # HRFormer-Base output channels
    num_keypoints: int = 17
    hidden_dim: int = 256
    
    # Loss settings
    use_target_weight: bool = True
    use_fusion_loss: bool = True
    
    # Fusion loss weights
    heatmap_loss_weight: float = 1.0
    offset_loss_weight: float = 1.0
    peak_loss_weight: float = 0.5
    variance_loss_weight: float = 0.1
    overlap_loss_weight: float = 0.05
    shape_loss_weight: float = 0.05
    
    # Gaussian heatmap settings
    target_sigma: float = 2.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training schedule
    max_epochs: int = 210
    val_interval: int = 10
    
    # Batch size
    batch_size: int = 32
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = 'AdamW'
    lr: float = 5e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate schedule
    warmup_epochs: int = 5
    warmup_lr: float = 5e-7
    lr_milestones: List[int] = field(default_factory=lambda: [170, 200])
    lr_gamma: float = 0.1
    
    # Augmentation
    flip_prob: float = 0.5
    half_body_prob: float = 0.3
    rotation_factor: float = 40.0
    scale_factor: Tuple[float, float] = (0.5, 1.5)
    
    # Checkpoint
    save_best: str = 'AP'
    checkpoint_dir: str = 'checkpoints/'
    
    # Device
    device: str = 'cuda'
    fp16: bool = True


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # Experiment
    exp_name: str = 'hrformer_base_coco_256x192'
    seed: int = 42


def get_config() -> Config:
    """Get default configuration."""
    return Config()
