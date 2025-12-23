"""
COCO Pose Dataset Implementation
"""

import os
import json
import copy
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

from .transforms import Compose, get_train_transforms, get_val_transforms


class COCOPoseDataset(Dataset):
    """COCO Keypoint Detection Dataset.
    
    Args:
        data_root: Root directory of the dataset.
        ann_file: Path to annotation file.
        img_prefix: Prefix for image paths.
        input_size: Input image size (width, height).
        heatmap_size: Heatmap size (width, height).
        sigma: Gaussian sigma for heatmap generation.
        num_keypoints: Number of keypoints.
        transforms: Data augmentation transforms.
        is_train: Whether in training mode.
    """
    
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        img_prefix: str,
        input_size: Tuple[int, int] = (192, 256),
        heatmap_size: Tuple[int, int] = (48, 64),
        sigma: float = 2.0,
        num_keypoints: int = 17,
        transforms: Optional[Compose] = None,
        is_train: bool = True,
        flip_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        self.data_root = data_root
        self.ann_file = os.path.join(data_root, ann_file)
        self.img_prefix = os.path.join(data_root, img_prefix)
        self.input_size = np.array(input_size)  # (w, h)
        self.heatmap_size = np.array(heatmap_size)  # (w, h)
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.transforms = transforms
        self.is_train = is_train
        self.flip_pairs = flip_pairs or [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
        ]
        
        # Load annotations
        self.coco = COCO(self.ann_file)
        self.db = self._load_annotations()
        
        print(f"Loaded {len(self.db)} samples from {self.ann_file}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from COCO format."""
        db = []
        
        for img_id in self.coco.getImgIds():
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                # Skip if no keypoints or invalid bbox
                if ann.get('num_keypoints', 0) == 0:
                    continue
                
                # Get bbox
                x, y, w, h = ann['bbox']
                if w <= 0 or h <= 0:
                    continue
                
                # Expand bbox
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_info['width'], x + w)
                y2 = min(img_info['height'], y + h)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Get keypoints
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                
                # Calculate center and scale
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                scale = np.array([x2 - x1, y2 - y1], dtype=np.float32)
                
                # Extend scale by 1.25
                scale = scale * 1.25
                
                db.append({
                    'image_file': os.path.join(self.img_prefix, img_info['file_name']),
                    'image_id': img_id,
                    'ann_id': ann['id'],
                    'center': center,
                    'scale': scale,
                    'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                    'keypoints': keypoints[:, :2].astype(np.float32),
                    'keypoints_visible': keypoints[:, 2].astype(np.float32),
                    'area': ann.get('area', w * h),
                })
        
        return db
    
    def __len__(self) -> int:
        return len(self.db)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        record = copy.deepcopy(self.db[idx])
        
        # Load image
        img = cv2.imread(record['image_file'])
        if img is None:
            raise ValueError(f"Failed to load image: {record['image_file']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare data dict
        data = {
            'img': img,
            'center': record['center'],
            'scale': record['scale'],
            'keypoints': record['keypoints'],
            'keypoints_visible': record['keypoints_visible'],
            'image_id': record['image_id'],
            'ann_id': record['ann_id'],
            'bbox': record['bbox'],
            'area': record['area'],
            'input_size': self.input_size,
            'heatmap_size': self.heatmap_size,
            'flip_pairs': self.flip_pairs,
        }
        
        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)
        
        # Generate heatmaps
        target, target_weight = self._generate_target(
            data['keypoints'], 
            data['keypoints_visible']
        )
        
        # Convert to tensor
        img_tensor = torch.from_numpy(data['img'].transpose(2, 0, 1)).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Keypoints in input image space (for fusion loss)
        keypoints_input = torch.from_numpy(data['keypoints']).float()
        keypoints_visible = torch.from_numpy(data['keypoints_visible']).float()
        
        return {
            'img': img_tensor,
            'target': torch.from_numpy(target).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'keypoints': keypoints_input,  # (K, 2) in input image space
            'keypoints_visible': keypoints_visible,  # (K,)
            'meta': {
                'image_id': data['image_id'],
                'ann_id': data['ann_id'],
                'center': data['center'],
                'scale': data['scale'],
                'bbox': data['bbox'],
                'area': data['area'],
            }
        }
    
    def _generate_target(
        self, 
        keypoints: np.ndarray, 
        keypoints_visible: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate target heatmaps.
        
        Args:
            keypoints: Keypoint coordinates (K, 2) in input image space.
            keypoints_visible: Visibility flags (K,).
            
        Returns:
            target: Heatmaps (K, H, W).
            target_weight: Weight for each keypoint (K, 1).
        """
        num_keypoints = self.num_keypoints
        heatmap_w, heatmap_h = self.heatmap_size
        input_w, input_h = self.input_size
        
        target = np.zeros((num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)
        target_weight = np.zeros((num_keypoints, 1), dtype=np.float32)
        
        # Scale factor from input to heatmap
        feat_stride = self.input_size / self.heatmap_size
        
        # Gaussian kernel size
        tmp_size = self.sigma * 3
        
        for joint_id in range(num_keypoints):
            target_weight[joint_id] = keypoints_visible[joint_id]
            
            if target_weight[joint_id] < 0.5:
                continue
            
            # Transform to heatmap coordinates
            mu_x = keypoints[joint_id, 0] / feat_stride[0]
            mu_y = keypoints[joint_id, 1] / feat_stride[1]
            
            # Check bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            
            if ul[0] >= heatmap_w or ul[1] >= heatmap_h or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue
            
            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
            
            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_h) - ul[1]
            
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_w)
            img_y = max(0, ul[1]), min(br[1], heatmap_h)
            
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return target, target_weight


def build_dataloader(
    cfg,
    is_train: bool = True,
) -> DataLoader:
    """Build dataloader.
    
    Args:
        cfg: Configuration object.
        is_train: Whether in training mode.
        
    Returns:
        DataLoader instance.
    """
    data_cfg = cfg.data
    train_cfg = cfg.train
    
    # Get transforms
    if is_train:
        transforms = get_train_transforms(
            input_size=data_cfg.input_size,
            flip_prob=train_cfg.flip_prob,
            rotation_factor=train_cfg.rotation_factor,
            scale_factor=train_cfg.scale_factor,
        )
        ann_file = data_cfg.train_ann
        img_prefix = data_cfg.train_img_prefix
    else:
        transforms = get_val_transforms(input_size=data_cfg.input_size)
        ann_file = data_cfg.val_ann
        img_prefix = data_cfg.val_img_prefix
    
    dataset = COCOPoseDataset(
        data_root=data_cfg.data_root,
        ann_file=ann_file,
        img_prefix=img_prefix,
        input_size=data_cfg.input_size,
        heatmap_size=data_cfg.heatmap_size,
        sigma=data_cfg.sigma,
        num_keypoints=data_cfg.num_keypoints,
        transforms=transforms,
        is_train=is_train,
        flip_pairs=data_cfg.flip_pairs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=is_train,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    
    return dataloader
