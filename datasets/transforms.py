"""
Data Augmentation Transforms for Pose Estimation
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, data: Dict) -> Dict:
        for t in self.transforms:
            data = t(data)
        return data


class TopdownAffine:
    """Apply affine transformation to crop and resize image.
    
    Args:
        input_size: Target input size (width, height).
    """
    
    def __init__(self, input_size: Tuple[int, int]):
        self.input_size = np.array(input_size)
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        center = data['center']
        scale = data['scale']
        keypoints = data['keypoints']
        
        # Get affine transform matrix
        trans = self._get_affine_transform(center, scale, self.input_size)
        
        # Apply to image
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )
        
        # Apply to keypoints
        for i in range(len(keypoints)):
            if data['keypoints_visible'][i] > 0:
                keypoints[i] = self._affine_transform(keypoints[i], trans)
        
        data['img'] = img
        data['keypoints'] = keypoints
        
        return data
    
    def _get_affine_transform(
        self, 
        center: np.ndarray, 
        scale: np.ndarray, 
        output_size: np.ndarray,
        rot: float = 0
    ) -> np.ndarray:
        """Get affine transform matrix."""
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]
        
        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
        
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        
        return trans
    
    def _get_dir(self, src_point: List, rot_rad: float) -> np.ndarray:
        """Get direction vector."""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        result = [0, 0]
        result[0] = src_point[0] * cs - src_point[1] * sn
        result[1] = src_point[0] * sn + src_point[1] * cs
        return np.array(result)
    
    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Get third point for affine transform."""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def _affine_transform(self, pt: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply affine transform to a point."""
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


class RandomFlip:
    """Random horizontal flip.
    
    Args:
        flip_prob: Probability of flipping.
    """
    
    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() < self.flip_prob:
            img = data['img']
            center = data['center']
            keypoints = data['keypoints']
            keypoints_visible = data['keypoints_visible']
            flip_pairs = data.get('flip_pairs', [])
            
            img_width = img.shape[1]
            
            # Flip image
            img = img[:, ::-1, :].copy()
            
            # Flip center
            center[0] = img_width - center[0] - 1
            
            # Flip keypoints
            keypoints[:, 0] = img_width - keypoints[:, 0] - 1
            
            # Swap left-right keypoints
            for pair in flip_pairs:
                keypoints[pair[0]], keypoints[pair[1]] = \
                    keypoints[pair[1]].copy(), keypoints[pair[0]].copy()
                keypoints_visible[pair[0]], keypoints_visible[pair[1]] = \
                    keypoints_visible[pair[1]], keypoints_visible[pair[0]]
            
            data['img'] = img
            data['center'] = center
            data['keypoints'] = keypoints
            data['keypoints_visible'] = keypoints_visible
        
        return data


class RandomBBoxTransform:
    """Random scale and rotation augmentation.
    
    Args:
        rotation_factor: Maximum rotation angle.
        scale_factor: Scale range (min, max).
    """
    
    def __init__(
        self,
        rotation_factor: float = 40.0,
        scale_factor: Tuple[float, float] = (0.5, 1.5),
        rotation_prob: float = 0.6,
    ):
        self.rotation_factor = rotation_factor
        self.scale_factor = scale_factor
        self.rotation_prob = rotation_prob
    
    def __call__(self, data: Dict) -> Dict:
        center = data['center']
        scale = data['scale']
        
        # Random scale
        scale_min, scale_max = self.scale_factor
        scale_factor = np.random.uniform(scale_min, scale_max)
        scale = scale * scale_factor
        
        # Random rotation
        if np.random.random() < self.rotation_prob:
            rotation = np.clip(
                np.random.randn() * self.rotation_factor,
                -self.rotation_factor * 2,
                self.rotation_factor * 2
            )
        else:
            rotation = 0
        
        data['center'] = center
        data['scale'] = scale
        data['rotation'] = rotation
        
        return data


class TopdownAffineWithRotation(TopdownAffine):
    """Affine transform with rotation support."""
    
    def __call__(self, data: Dict) -> Dict:
        img = data['img']
        center = data['center']
        scale = data['scale']
        keypoints = data['keypoints']
        rotation = data.get('rotation', 0)
        
        # Get affine transform matrix with rotation
        trans = self._get_affine_transform(center, scale, self.input_size, rotation)
        
        # Apply to image
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )
        
        # Apply to keypoints
        for i in range(len(keypoints)):
            if data['keypoints_visible'][i] > 0:
                keypoints[i] = self._affine_transform(keypoints[i], trans)
                
                # Check if keypoint is still visible
                if (keypoints[i, 0] < 0 or keypoints[i, 0] >= self.input_size[0] or
                    keypoints[i, 1] < 0 or keypoints[i, 1] >= self.input_size[1]):
                    data['keypoints_visible'][i] = 0
        
        data['img'] = img
        data['keypoints'] = keypoints
        
        return data


class RandomHalfBody:
    """Random half body augmentation.
    
    Args:
        prob: Probability of applying half body augmentation.
        min_keypoints: Minimum visible keypoints required.
    """
    
    # Upper body and lower body keypoint indices (COCO format)
    UPPER_BODY_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    LOWER_BODY_IDS = [11, 12, 13, 14, 15, 16]
    
    def __init__(self, prob: float = 0.3, min_keypoints: int = 3):
        self.prob = prob
        self.min_keypoints = min_keypoints
    
    def __call__(self, data: Dict) -> Dict:
        if np.random.random() > self.prob:
            return data
        
        keypoints = data['keypoints']
        keypoints_visible = data['keypoints_visible']
        
        # Get upper and lower body keypoints
        upper_kpts = []
        lower_kpts = []
        
        for i in self.UPPER_BODY_IDS:
            if keypoints_visible[i] > 0:
                upper_kpts.append(keypoints[i])
        
        for i in self.LOWER_BODY_IDS:
            if keypoints_visible[i] > 0:
                lower_kpts.append(keypoints[i])
        
        # Choose half body
        if len(upper_kpts) >= self.min_keypoints and len(lower_kpts) >= self.min_keypoints:
            selected_kpts = upper_kpts if np.random.random() < 0.5 else lower_kpts
        elif len(upper_kpts) >= self.min_keypoints:
            selected_kpts = upper_kpts
        elif len(lower_kpts) >= self.min_keypoints:
            selected_kpts = lower_kpts
        else:
            return data
        
        selected_kpts = np.array(selected_kpts)
        
        # Update center and scale
        center = selected_kpts.mean(axis=0)
        
        x_min, y_min = selected_kpts.min(axis=0)
        x_max, y_max = selected_kpts.max(axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        scale = np.array([w, h]) * 1.5
        scale = np.maximum(scale, data['scale'] * 0.5)
        
        data['center'] = center
        data['scale'] = scale
        
        return data


def get_train_transforms(
    input_size: Tuple[int, int],
    flip_prob: float = 0.5,
    rotation_factor: float = 40.0,
    scale_factor: Tuple[float, float] = (0.5, 1.5),
) -> Compose:
    """Get training transforms."""
    return Compose([
        RandomFlip(flip_prob=flip_prob),
        RandomHalfBody(prob=0.3),
        RandomBBoxTransform(
            rotation_factor=rotation_factor,
            scale_factor=scale_factor,
        ),
        TopdownAffineWithRotation(input_size=input_size),
    ])


def get_val_transforms(input_size: Tuple[int, int]) -> Compose:
    """Get validation transforms."""
    return Compose([
        TopdownAffine(input_size=input_size),
    ])
