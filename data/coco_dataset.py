"""
COCO Format Dataset Loader for Preterm Infant Pose Estimation
Handles data loading, augmentation, and heatmap generation
"""
import os
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as transforms


class PreemieCocoDataset(Dataset):
    """
    Dataset class for preterm infant pose estimation
    Follows COCO keypoint format
    """
    def __init__(self, config, ann_file, img_dir, is_train=True):
        self.config = config
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.is_train = is_train
        
        # COCO API
        self.coco = COCO(ann_file)
        
        # Get image ids with keypoint annotations
        self.img_ids = list(self.coco.imgs.keys())
        
        # Filter out images without annotations
        self.img_ids = [
            img_id for img_id in self.img_ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
        ]
        
        # Dataset settings
        self.num_joints = config.MODEL.NUM_JOINTS
        self.image_size = config.MODEL.IMAGE_SIZE
        self.heatmap_size = config.MODEL.HEATMAP_SIZE
        self.sigma = config.MODEL.SIGMA if hasattr(config.MODEL, 'SIGMA') else 2
        
        # Data augmentation
        self.transform = self._build_transforms(is_train)
        
        print(f'Loaded {len(self.img_ids)} images from {ann_file}')
    
    def _build_transforms(self, is_train):
        """Build data augmentation pipeline"""
        if is_train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary containing:
            - image: (3, H, W) normalized image tensor
            - target_heatmap: (K, H, W) ground truth heatmaps
            - target_coords: (K, 2) normalized coordinates
            - target_weight: (K, 1) visibility flags
            - bbox: (4,) bounding box [x, y, w, h]
            - center: (2,) center of bounding box
            - scale: (2,) scale factors
            - image_id: image ID
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Use first person annotation (assuming one infant per image)
        ann = anns[0]
        
        # Get bounding box
        bbox = np.array(ann['bbox'], dtype=np.float32)  # [x, y, w, h]
        
        # Get keypoints [x1, y1, v1, x2, y2, v2, ...]
        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        joints = keypoints[:, :2]  # (K, 2)
        joints_vis = keypoints[:, 2:3]  # (K, 1)
        
        # Calculate center and scale from bbox
        center = np.array([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5], dtype=np.float32)
        scale = np.array([bbox[2], bbox[3]], dtype=np.float32)
        
        # Apply augmentation (rotation, scaling) if training
        if self.is_train:
            image, joints, joints_vis = self._augment_data(image, joints, joints_vis, center, scale)
        
        # Crop and resize image
        image_resized = self._crop_and_resize(image, center, scale, self.image_size)
        
        # Transform joints to resized image coordinates
        joints_resized = self._transform_joints(joints, center, scale, self.image_size)
        
        # Generate heatmaps
        target_heatmap, target_weight = self._generate_heatmaps(
            joints_resized, joints_vis, self.heatmap_size
        )
        
        # Normalize coordinates to [0, 1]
        target_coords = joints_resized / np.array(self.image_size)
        
        # Apply image transforms (normalization)
        image_tensor = self.transform(image_resized)
        
        return {
            'image': image_tensor,
            'target_heatmap': torch.from_numpy(target_heatmap).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'bbox': torch.from_numpy(bbox).float(),
            'center': torch.from_numpy(center).float(),
            'scale': torch.from_numpy(scale).float(),
            'image_id': torch.tensor(img_id).long()
        }
    
    def _augment_data(self, image, joints, joints_vis, center, scale):
        """Apply random augmentation for training"""
        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-30, 30)
            image, joints = self._rotate(image, joints, center, angle)
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            scale = scale * scale_factor
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            joints[:, 0] = image.shape[1] - joints[:, 0]
            # Swap left-right keypoints
            joints, joints_vis = self._flip_joints(joints, joints_vis)
        
        return image, joints, joints_vis
    
    def _rotate(self, image, joints, center, angle):
        """Rotate image and joints"""
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Transform joints
        joints_homo = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = (M @ joints_homo.T).T
        
        return image, joints
    
    def _flip_joints(self, joints, joints_vis):
        """Swap left-right joints for horizontal flip"""
        # Define left-right pairs (adjust based on your keypoint definition)
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        
        for left, right in pairs:
            joints[[left, right]] = joints[[right, left]]
            joints_vis[[left, right]] = joints_vis[[right, left]]
        
        return joints, joints_vis
    
    def _crop_and_resize(self, image, center, scale, output_size):
        """Crop image based on bounding box and resize"""
        h, w = image.shape[:2]
        
        # Calculate crop box with padding
        padding = 0.25
        x1 = int(max(0, center[0] - scale[0] * (1 + padding) / 2))
        y1 = int(max(0, center[1] - scale[1] * (1 + padding) / 2))
        x2 = int(min(w, center[0] + scale[0] * (1 + padding) / 2))
        y2 = int(min(h, center[1] + scale[1] * (1 + padding) / 2))
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Resize
        resized = cv2.resize(cropped, tuple(output_size))
        
        return resized
    
    def _transform_joints(self, joints, center, scale, output_size):
        """Transform joint coordinates to cropped and resized image space"""
        padding = 0.25
        
        # Calculate crop offset
        x_offset = center[0] - scale[0] * (1 + padding) / 2
        y_offset = center[1] - scale[1] * (1 + padding) / 2
        
        # Transform coordinates
        joints_transformed = joints.copy()
        joints_transformed[:, 0] = (joints[:, 0] - x_offset) / (scale[0] * (1 + padding)) * output_size[0]
        joints_transformed[:, 1] = (joints[:, 1] - y_offset) / (scale[1] * (1 + padding)) * output_size[1]
        
        return joints_transformed
    
    def _generate_heatmaps(self, joints, joints_vis, heatmap_size):
        """
        Generate Gaussian heatmaps for each keypoint
        
        Args:
            joints: (K, 2) joint coordinates in image space
            joints_vis: (K, 1) visibility flags
            heatmap_size: (H, W) output heatmap size
        """
        num_joints = self.num_joints
        target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        
        # Scale joints to heatmap size
        scale_x = heatmap_size[1] / self.image_size[0]
        scale_y = heatmap_size[0] / self.image_size[1]
        
        for joint_id in range(num_joints):
            vis = joints_vis[joint_id, 0]
            
            if vis > 0:  # Visible joint
                target_weight[joint_id] = 1.0
                
                # Joint position in heatmap coordinates
                mu_x = joints[joint_id, 0] * scale_x
                mu_y = joints[joint_id, 1] * scale_y
                
                # Check if joint is within heatmap bounds
                if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_size[1] or mu_y >= heatmap_size[0]:
                    target_weight[joint_id] = 0.0
                    continue
                
                # Generate Gaussian heatmap
                tmp_size = self.sigma * 3
                
                # Upper left and lower right coordinates
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                
                # Clip to heatmap boundaries
                ul[0] = max(0, ul[0])
                ul[1] = max(0, ul[1])
                br[0] = min(heatmap_size[1], br[0])
                br[1] = min(heatmap_size[0], br[1])
                
                # Generate Gaussian kernel
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                
                # Gaussian formula
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
                
                # Extract valid region
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
                
                img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[0])
                
                # Place Gaussian on heatmap
                target[joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return target, target_weight


def build_dataloader(config, is_train=True):
    """Factory function to build data loader"""
    if is_train:
        dataset = PreemieCocoDataset(
            config,
            ann_file=os.path.join(config.DATA.DATA_DIR, 'annotations/train.json'),
            img_dir=os.path.join(config.DATA.DATA_DIR, 'images/train'),
            is_train=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=config.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
    else:
        dataset = PreemieCocoDataset(
            config,
            ann_file=os.path.join(config.DATA.DATA_DIR, 'annotations/val.json'),
            img_dir=os.path.join(config.DATA.DATA_DIR, 'images/val'),
            is_train=False
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.TEST.NUM_WORKERS,
            pin_memory=True
        )
    
    return dataloader
