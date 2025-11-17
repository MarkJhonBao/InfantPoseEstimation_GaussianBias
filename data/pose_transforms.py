import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2


class LoadImage:
    """加载图像"""
    
    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32
    
    def __call__(self, results: Dict) -> Dict:
        """
        Args:
            results: dict with 'img_path' key
        Returns:
            results: dict with 'img' and 'img_shape' keys added
        """
        img_path = results['img_path']
        
        # 使用cv2加载图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.to_float32:
            img = img.astype(np.float32)
        
        results['img'] = img
        results['img_shape'] = img.shape[:2]  # (H, W)
        results['ori_shape'] = img.shape[:2]
        
        return results


class GetBBoxCenterScale:
    """从边界框获取中心点和尺度"""
    
    def __init__(self, padding: float = 1.25):
        self.padding = padding
    
    def __call__(self, results: Dict) -> Dict:
        """
        Args:
            results: dict with 'bbox' key [x, y, w, h] or [x1, y1, x2, y2]
        Returns:
            results: dict with 'center' and 'scale' keys added
        """
        bbox = np.array(results['bbox']).astype(np.float32)
        
        if len(bbox) == 4:
            # 假设格式是 [x, y, w, h]
            if bbox[2] < bbox[0]:  # 如果是 [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                x, y = x1, y1
            else:
                x, y, w, h = bbox
        else:
            raise ValueError(f"Unsupported bbox format with length {len(bbox)}")
        
        # 计算中心点
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        
        # 计算尺度（考虑padding）
        scale = np.array([w, h], dtype=np.float32) * self.padding
        
        results['center'] = center
        results['scale'] = scale
        results['bbox'] = np.array([x, y, w, h], dtype=np.float32)
        
        return results


class RandomFlip:
    """随机翻转图像和关键点"""
    
    def __init__(self, direction: str = 'horizontal', prob: float = 0.5):
        self.direction = direction
        self.prob = prob
        assert direction in ['horizontal', 'vertical']
    
    def __call__(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results
        
        img = results['img']
        h, w = img.shape[:2]
        
        if self.direction == 'horizontal':
            # 水平翻转
            results['img'] = np.fliplr(img).copy()
            
            # 翻转中心点
            if 'center' in results:
                center = results['center'].copy()
                center[0] = w - center[0]
                results['center'] = center
            
            # 翻转关键点
            if 'keypoints' in results:
                keypoints = results['keypoints'].copy()
                keypoints[..., 0] = w - keypoints[..., 0]
                
                # 交换左右关键点（需要flip_pairs）
                if 'flip_pairs' in results:
                    for left_idx, right_idx in results['flip_pairs']:
                        keypoints[[left_idx, right_idx]] = keypoints[[right_idx, left_idx]]
                
                results['keypoints'] = keypoints
            
            results['flipped'] = True
        
        elif self.direction == 'vertical':
            # 垂直翻转
            results['img'] = np.flipud(img).copy()
            
            if 'center' in results:
                center = results['center'].copy()
                center[1] = h - center[1]
                results['center'] = center
            
            if 'keypoints' in results:
                keypoints = results['keypoints'].copy()
                keypoints[..., 1] = h - keypoints[..., 1]
                results['keypoints'] = keypoints
            
            results['flipped'] = True
        
        return results


class RandomHalfBody:
    """随机半身增强，用于人体姿态估计"""
    
    def __init__(self, 
                 min_total_keypoints: int = 8,
                 min_half_keypoints: int = 2,
                 prob: float = 0.3,
                 upper_body_ids: Optional[List[int]] = None,
                 lower_body_ids: Optional[List[int]] = None):
        self.min_total_keypoints = min_total_keypoints
        self.min_half_keypoints = min_half_keypoints
        self.prob = prob
        
        # COCO默认的上下半身关键点索引
        if upper_body_ids is None:
            # 鼻子, 眼睛, 耳朵, 肩膀, 肘部, 手腕
            self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.upper_body_ids = upper_body_ids
        
        if lower_body_ids is None:
            # 臀部, 膝盖, 脚踝
            self.lower_body_ids = [11, 12, 13, 14, 15, 16]
        else:
            self.lower_body_ids = lower_body_ids
    
    def __call__(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results
        
        if 'keypoints' not in results or 'keypoints_visible' not in results:
            return results
        
        keypoints = results['keypoints']  # shape: (num_keypoints, 2 or 3)
        keypoints_visible = results['keypoints_visible']  # shape: (num_keypoints,)
        
        # 检查可见关键点数量
        num_visible = np.sum(keypoints_visible > 0)
        if num_visible < self.min_total_keypoints:
            return results
        
        # 随机选择上半身或下半身
        if np.random.rand() < 0.5:
            selected_ids = self.upper_body_ids
        else:
            selected_ids = self.lower_body_ids
        
        # 获取选中部位的可见关键点
        selected_visible = keypoints_visible[selected_ids] > 0
        if np.sum(selected_visible) < self.min_half_keypoints:
            return results
        
        # 获取选中关键点的坐标
        selected_keypoints = keypoints[selected_ids][selected_visible]
        
        if len(selected_keypoints) < self.min_half_keypoints:
            return results
        
        # 计算新的bbox
        x_coords = selected_keypoints[:, 0]
        y_coords = selected_keypoints[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # 添加一些边距
        margin = 0.15
        x_min = max(0, x_min - w * margin)
        y_min = max(0, y_min - h * margin)
        x_max = min(results['img'].shape[1], x_max + w * margin)
        y_max = min(results['img'].shape[0], y_max + h * margin)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # 更新center和scale
        results['center'] = np.array([x_min + w * 0.5, y_min + h * 0.5], dtype=np.float32)
        results['scale'] = np.array([w, h], dtype=np.float32) * 1.25
        
        return results


class RandomBBoxTransform:
    """随机变换边界框（缩放和平移）"""
    
    def __init__(self,
                 scale_factor: Tuple[float, float] = (0.75, 1.5),
                 shift_factor: float = 0.16,
                 rotate_factor: float = 40,
                 prob: float = 1.0):
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.rotate_factor = rotate_factor
        self.prob = prob
    
    def __call__(self, results: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return results
        
        if 'center' not in results or 'scale' not in results:
            return results
        
        center = results['center'].copy()
        scale = results['scale'].copy()
        
        # 随机缩放
        scale_factor = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
        scale = scale * scale_factor
        
        # 随机平移
        shift_x = np.random.uniform(-self.shift_factor, self.shift_factor) * scale[0]
        shift_y = np.random.uniform(-self.shift_factor, self.shift_factor) * scale[1]
        center[0] += shift_x
        center[1] += shift_y
        
        # 随机旋转
        rotation = np.random.uniform(-self.rotate_factor, self.rotate_factor)
        
        results['center'] = center
        results['scale'] = scale
        results['rotation'] = rotation
        
        return results


class TopdownAffine:
    """自顶向下的仿射变换"""
    
    def __init__(self, input_size: Tuple[int, int]):
        self.input_size = input_size  # (W, H)
    
    def _get_affine_matrix(self,
                          center: np.ndarray,
                          scale: np.ndarray,
                          output_size: Tuple[int, int],
                          rotation: float = 0.0) -> np.ndarray:
        """获取仿射变换矩阵"""
        
        # 输出尺寸
        w, h = output_size
        
        # 计算从原图到目标图的变换
        scale_x = scale[0] / w
        scale_y = scale[1] / h
        
        # 构建变换矩阵
        # 1. 平移到原点
        # 2. 旋转
        # 3. 缩放
        # 4. 平移到目标中心
        
        rot_rad = np.deg2rad(rotation)
        cos_val = np.cos(rot_rad)
        sin_val = np.sin(rot_rad)
        
        # 源图像的三个点
        src_w = scale[0]
        src_h = scale[1]
        src_center = center
        
        # 目标图像的三个点
        dst_w = w
        dst_h = h
        dst_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        
        # 源图像的三个关键点
        src_pts = np.float32([
            src_center,
            src_center + np.array([0, src_h * -0.5]),
            src_center + np.array([src_w * 0.5, src_h * -0.5])
        ])
        
        # 应用旋转
        if rotation != 0:
            rotation_matrix = np.array([
                [cos_val, sin_val],
                [-sin_val, cos_val]
            ], dtype=np.float32)
            
            for i in range(1, 3):
                src_pts[i] = src_center + rotation_matrix @ (src_pts[i] - src_center)
        
        # 目标图像的三个关键点
        dst_pts = np.float32([
            dst_center,
            dst_center + np.array([0, dst_h * -0.5]),
            dst_center + np.array([dst_w * 0.5, dst_h * -0.5])
        ])
        
        # 计算仿射变换矩阵
        affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        
        return affine_matrix
    
    def __call__(self, results: Dict) -> Dict:
        img = results['img']
        center = results['center']
        scale = results['scale']
        rotation = results.get('rotation', 0.0)
        
        # 获取仿射变换矩阵
        affine_matrix = self._get_affine_matrix(
            center, scale, self.input_size, rotation
        )
        
        # 应用仿射变换到图像
        transformed_img = cv2.warpAffine(
            img,
            affine_matrix,
            self.input_size,
            flags=cv2.INTER_LINEAR
        )
        
        results['img'] = transformed_img
        results['img_shape'] = transformed_img.shape[:2]
        
        # 变换关键点
        if 'keypoints' in results:
            keypoints = results['keypoints'].copy()  # (num_keypoints, 2 or 3)
            num_keypoints = keypoints.shape[0]
            
            # 应用仿射变换
            keypoints_homo = np.concatenate([
                keypoints[:, :2],
                np.ones((num_keypoints, 1))
            ], axis=1)  # (num_keypoints, 3)
            
            transformed_keypoints = keypoints_homo @ affine_matrix.T
            
            # 更新关键点
            if keypoints.shape[1] == 2:
                results['keypoints'] = transformed_keypoints
            else:  # 保留第三维（可见性或深度）
                results['keypoints'] = np.concatenate([
                    transformed_keypoints,
                    keypoints[:, 2:3]
                ], axis=1)
        
        results['affine_matrix'] = affine_matrix
        
        return results


class GenerateTarget:
    """生成训练目标（关键点热图）"""
    
    def __init__(self, encoder: Dict):
        self.encoder = encoder
        self.input_size = encoder.get('input_size', (256, 256))
        self.heatmap_size = encoder.get('heatmap_size', (64, 64))
        self.sigma = encoder.get('sigma', 2.0)
        self.use_udp = encoder.get('use_udp', False)
    
    def _generate_gaussian_heatmap(self,
                                   heatmap: np.ndarray,
                                   center: np.ndarray,
                                   sigma: float) -> np.ndarray:
        """生成高斯热图"""
        h, w = heatmap.shape
        
        # 生成网格
        x = np.arange(0, w, 1, dtype=np.float32)
        y = np.arange(0, h, 1, dtype=np.float32)
        y = y[:, np.newaxis]
        
        # 计算高斯分布
        x0, y0 = center
        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        # 叠加到热图上
        heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap
    
    def __call__(self, results: Dict) -> Dict:
        if 'keypoints' not in results:
            return results
        
        keypoints = results['keypoints']  # (num_keypoints, 2 or 3)
        keypoints_visible = results.get('keypoints_visible', 
                                       np.ones(len(keypoints)))
        
        num_keypoints = keypoints.shape[0]
        heatmap_h, heatmap_w = self.heatmap_size
        input_h, input_w = self.input_size
        
        # 缩放关键点到热图尺寸
        scale_w = heatmap_w / input_w
        scale_h = heatmap_h / input_h
        
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] *= scale_w
        scaled_keypoints[:, 1] *= scale_h
        
        # 生成热图
        heatmaps = np.zeros((num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)
        keypoint_weights = np.ones(num_keypoints, dtype=np.float32)
        
        for i in range(num_keypoints):
            if keypoints_visible[i] > 0:
                center = scaled_keypoints[i, :2]
                
                # 检查关键点是否在图像范围内
                if 0 <= center[0] < heatmap_w and 0 <= center[1] < heatmap_h:
                    heatmaps[i] = self._generate_gaussian_heatmap(
                        heatmaps[i], center, self.sigma
                    )
                else:
                    keypoint_weights[i] = 0.0
            else:
                keypoint_weights[i] = 0.0
        
        results['heatmaps'] = heatmaps
        results['keypoint_weights'] = keypoint_weights
        
        return results


class PackPoseInputs:
    """打包输入数据"""
    
    def __init__(self, meta_keys: Optional[List[str]] = None):
        if meta_keys is None:
            self.meta_keys = [
                'img_path', 'ori_shape', 'img_shape', 'input_size',
                'center', 'scale', 'flip', 'flip_direction'
            ]
        else:
            self.meta_keys = meta_keys
    
    def __call__(self, results: Dict) -> Dict:
        packed_results = {}
        
        # 图像转为tensor (H, W, C) -> (C, H, W)
        img = results['img']
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        packed_results['img'] = img_tensor
        
        # 热图转为tensor
        if 'heatmaps' in results:
            heatmaps = torch.from_numpy(results['heatmaps'])
            packed_results['heatmaps'] = heatmaps
        
        # 关键点权重
        if 'keypoint_weights' in results:
            keypoint_weights = torch.from_numpy(results['keypoint_weights'])
            packed_results['keypoint_weights'] = keypoint_weights
        
        # 关键点坐标
        if 'keypoints' in results:
            keypoints = torch.from_numpy(results['keypoints'])
            packed_results['keypoints'] = keypoints
        
        # 元信息
        data_sample = {}
        for key in self.meta_keys:
            if key in results:
                data_sample[key] = results[key]
        
        packed_results['data_sample'] = data_sample
        
        return packed_results


def build_train_pipeline(codec: Dict, flip_pairs: Optional[List[Tuple[int, int]]] = None):
    """构建训练pipeline"""
    pipeline = [
        LoadImage(),
        GetBBoxCenterScale(),
        RandomFlip(direction='horizontal', prob=0.5),
        RandomHalfBody(),
        RandomBBoxTransform(),
        TopdownAffine(input_size=codec['input_size']),
        GenerateTarget(encoder=codec),
        PackPoseInputs()
    ]
    return pipeline


def build_val_pipeline(codec: Dict):
    """构建验证pipeline"""
    pipeline = [
        LoadImage(),
        GetBBoxCenterScale(),
        TopdownAffine(input_size=codec['input_size']),
        PackPoseInputs()
    ]
    return pipeline


# 示例使用
if __name__ == '__main__':
    # 配置编码器
    codec = {
        'input_size': (256, 192),  # (W, H)
        'heatmap_size': (64, 48),  # (W, H)
        'sigma': 2.0
    }
    
    # COCO flip pairs (左右对称的关键点对)
    flip_pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    
    # 构建pipeline
    train_pipeline = build_train_pipeline(codec, flip_pairs)
    val_pipeline = build_val_pipeline(codec)
    
    # 模拟输入数据
    results = {
        'img_path': 'path/to/image.jpg',
        'bbox': [100, 100, 200, 300],  # [x, y, w, h]
        'keypoints': np.random.rand(17, 2) * 100 + 100,  # COCO 17个关键点
        'keypoints_visible': np.ones(17),
        'flip_pairs': flip_pairs
    }
    
    # 执行训练pipeline
    print("执行训练pipeline...")
    for i, transform in enumerate(train_pipeline):
        print(f"Step {i+1}: {transform.__class__.__name__}")
        results = transform(results)
    
    print(f"\n最终输出:")
    print(f"- Image shape: {results['img'].shape}")
    if 'heatmaps' in results:
        print(f"- Heatmaps shape: {results['heatmaps'].shape}")
    if 'keypoint_weights' in results:
        print(f"- Keypoint weights shape: {results['keypoint_weights'].shape}")
