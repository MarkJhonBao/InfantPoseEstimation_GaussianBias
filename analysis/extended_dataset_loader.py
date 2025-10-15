"""
支持扩展关键点的COCO数据集加载器
可处理任意数量的关键点（13点、17点、68点、127点等）
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pycocotools.coco import COCO
import torchvision.transforms as transforms


class ExtendedCocoDataset(Dataset):
    """
    扩展的COCO关键点数据集加载器
    支持任意数量的关键点
    """
    
    def __init__(self, config, ann_file, img_dir, is_train=True, 
                 keypoint_groups=None):
        """
        Args:
            config: 配置对象
            ann_file: COCO标注文件路径
            img_dir: 图像目录
            is_train: 是否为训练模式
            keypoint_groups: 关键点分组（用于多任务学习）
                例如: {'body': [0, 16], 'face': [17, 84], 'hands': [85, 126]}
        """
        self.config = config
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.is_train = is_train
        
        # 加载COCO数据
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        
        # 过滤无标注的图像
        self.img_ids = [
            img_id for img_id in self.img_ids
            if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
        ]
        
        # 获取关键点配置
        cat = self.coco.loadCats(self.coco.getCatIds())[0]
        self.num_joints = len(cat['keypoints'])
        self.keypoint_names = cat['keypoints']
        self.skeleton = cat.get('skeleton', [])
        
        # 关键点分组（用于多任务学习）
        self.keypoint_groups = keypoint_groups
        
        # 图像大小和热图大小
        self.image_size = config.MODEL.IMAGE_SIZE
        self.heatmap_size = config.MODEL.HEATMAP_SIZE
        self.sigma = config.MODEL.SIGMA if hasattr(config.MODEL, 'SIGMA') else 2
        
        # 数据增强
        self.transform = self._build_transforms(is_train)
        
        print(f'ExtendedCocoDataset初始化:')
        print(f'  图像数: {len(self.img_ids)}')
        print(f'  关键点数: {self.num_joints}')
        print(f'  关键点名称: {self.keypoint_names[:5]}... (showing first 5)')
        if self.keypoint_groups:
            print(f'  关键点分组: {list(self.keypoint_groups.keys())}')
    
    def _build_transforms(self, is_train):
        """构建数据增强"""
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
        返回:
            Dictionary包含:
            - image: (3, H, W) 图像
            - target_heatmap: (K, H, W) 热图
            - target_coords: (K, 2) 坐标
            - target_weight: (K, 1) 可见性权重
            
            如果有分组:
            - target_heatmap_groups: dict with group heatmaps
            - target_coords_groups: dict with group coords
        """
        img_id = self.img_ids[idx]
        
        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        ann = anns[0]  # 假设每张图一个人
        
        # 获取边界框和关键点
        bbox = np.array(ann['bbox'], dtype=np.float32)
        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        joints = keypoints[:, :2]
        joints_vis = keypoints[:, 2:3]
        
        # 计算中心和尺度
        center = np.array([bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5])
        scale = np.array([bbox[2], bbox[3]])
        
        # 数据增强
        if self.is_train:
            image, joints, joints_vis = self._augment_data(
                image, joints, joints_vis, center, scale
            )
        
        # 裁剪和调整大小
        image_resized = self._crop_and_resize(image, center, scale, self.image_size)
        
        # 转换关键点坐标到调整后的图像空间
        joints_resized = self._transform_joints(joints, center, scale, self.image_size)
        
        # 生成热图
        target_heatmap, target_weight = self._generate_heatmaps(
            joints_resized, joints_vis, self.heatmap_size
        )
        
        # 归一化坐标
        target_coords = joints_resized / np.array(self.image_size)
        
        # 应用图像变换
        image_tensor = self.transform(image_resized)
        
        result = {
            'image': image_tensor,
            'target_heatmap': torch.from_numpy(target_heatmap).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'bbox': torch.from_numpy(bbox).float(),
            'center': torch.from_numpy(center).float(),
            'scale': torch.from_numpy(scale).float(),
            'image_id': torch.tensor(img_id).long(),
            'num_joints': self.num_joints
        }
        
        # 如果有分组，生成分组数据
        if self.keypoint_groups:
            result['groups'] = {}
            for group_name, (start_idx, end_idx) in self.keypoint_groups.items():
                result['groups'][group_name] = {
                    'heatmap': result['target_heatmap'][start_idx:end_idx+1],
                    'coords': result['target_coords'][start_idx:end_idx+1],
                    'weight': result['target_weight'][start_idx:end_idx+1]
                }
        
        return result
    
    def _augment_data(self, image, joints, joints_vis, center, scale):
        """数据增强"""
        # 随机旋转
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-30, 30)
            image, joints = self._rotate(image, joints, center, angle)
        
        # 随机缩放
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            scale = scale * scale_factor
        
        # 随机水平翻转
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            joints[:, 0] = image.shape[1] - joints[:, 0]
            joints, joints_vis = self._flip_joints(joints, joints_vis)
        
        return image, joints, joints_vis
    
    def _flip_joints(self, joints, joints_vis):
        """翻转时交换左右关键点"""
        # 自动检测左右对
        left_right_pairs = []
        for i, name in enumerate(self.keypoint_names):
            if 'left' in name.lower():
                right_name = name.lower().replace('left', 'right')
                for j, other_name in enumerate(self.keypoint_names):
                    if other_name.lower() == right_name:
                        left_right_pairs.append((i, j))
                        break
        
        # 交换
        for left, right in left_right_pairs:
            joints[[left, right]] = joints[[right, left]]
            joints_vis[[left, right]] = joints_vis[[right, left]]
        
        return joints, joints_vis
    
    def _rotate(self, image, joints, center, angle):
        """旋转图像和关键点"""
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # 转换关键点
        joints_homo = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = (M @ joints_homo.T).T
        
        return image, joints
    
    def _crop_and_resize(self, image, center, scale, output_size):
        """裁剪并调整图像大小"""
        h, w = image.shape[:2]
        
        padding = 0.25
        x1 = int(max(0, center[0] - scale[0] * (1 + padding) / 2))
        y1 = int(max(0, center[1] - scale[1] * (1 + padding) / 2))
        x2 = int(min(w, center[0] + scale[0] * (1 + padding) / 2))
        y2 = int(min(h, center[1] + scale[1] * (1 + padding) / 2))
        
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, tuple(output_size))
        
        return resized
    
    def _transform_joints(self, joints, center, scale, output_size):
        """转换关键点坐标"""
        padding = 0.25
        
        x_offset = center[0] - scale[0] * (1 + padding) / 2
        y_offset = center[1] - scale[1] * (1 + padding) / 2
        
        joints_transformed = joints.copy()
        joints_transformed[:, 0] = (joints[:, 0] - x_offset) / (scale[0] * (1 + padding)) * output_size[0]
        joints_transformed[:, 1] = (joints[:, 1] - y_offset) / (scale[1] * (1 + padding)) * output_size[1]
        
        return joints_transformed
    
    def _generate_heatmaps(self, joints, joints_vis, heatmap_size):
        """生成高斯热图"""
        target = np.zeros((self.num_joints, heatmap_size[0], heatmap_size[1]), 
                         dtype=np.float32)
        target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)
        
        scale_x = heatmap_size[1] / self.image_size[0]
        scale_y = heatmap_size[0] / self.image_size[1]
        
        for joint_id in range(self.num_joints):
            vis = joints_vis[joint_id, 0]
            
            if vis > 0:
                target_weight[joint_id] = 1.0
                
                mu_x = joints[joint_id, 0] * scale_x
                mu_y = joints[joint_id, 1] * scale_y
                
                if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_size[1] or mu_y >= heatmap_size[0]:
                    target_weight[joint_id] = 0.0
                    continue
                
                # 生成高斯热图
                tmp_size = self.sigma * 3
                
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                
                ul[0] = max(0, ul[0])
                ul[1] = max(0, ul[1])
                br[0] = min(heatmap_size[1], br[0])
                br[1] = min(heatmap_size[0], br[1])
                
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
                
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
                
                img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[0])
                
                target[joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return target, target_weight


class MultiTaskKeypointDataset(ExtendedCocoDataset):
    """
    多任务关键点数据集
    可同时训练多个部位（身体、面部、手部等）
    """
    
    def __init__(self, config, ann_file, img_dir, is_train=True):
        # 定义关键点分组
        # 假设: 0-16身体, 17-84面部, 85-105左手, 106-126右手
        keypoint_groups = {
            'body': (0, 16),
            'face': (17, 84),
            'left_hand': (85, 105),
            'right_hand': (106, 126)
        }
        
        super().__init__(config, ann_file, img_dir, is_train, keypoint_groups)
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        
        # 为每个任务准备数据
        result['task_data'] = {}
        for task_name, group_data in result['groups'].items():
            result['task_data'][task_name] = {
                'heatmap': group_data['heatmap'],
                'coords': group_data['coords'],
                'weight': group_data['weight'],
                'num_joints': group_data['heatmap'].shape[0]
            }
        
        return result


# 工具函数
def get_keypoint_group_names(ann_file):
    """自动检测关键点分组"""
    coco = COCO(ann_file)
    cat = coco.loadCats(coco.getCatIds())[0]
    keypoint_names = cat['keypoints']
    
    # 简单的启发式分组
    groups = {}
    
    # 检测身体关键点
    body_keywords = ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'eye', 'ear', 'nose']
    body_indices = [i for i, name in enumerate(keypoint_names) 
                   if any(kw in name.lower() for kw in body_keywords)]
    if body_indices:
        groups['body'] = (min(body_indices), max(body_indices))
    
    # 检测面部关键点
    face_keywords = ['jaw', 'eyebrow', 'nose_', 'eye_', 'lip']
    face_indices = [i for i, name in enumerate(keypoint_names) 
                   if any(kw in name.lower() for kw in face_keywords)]
    if face_indices:
        groups['face'] = (min(face_indices), max(face_indices))
    
    # 检测手部关键点
    hand_keywords = ['thumb', 'index', 'middle', 'ring', 'pinky']
    left_hand_indices = [i for i, name in enumerate(keypoint_names) 
                        if 'left' in name.lower() and any(kw in name.lower() for kw in hand_keywords)]
    right_hand_indices = [i for i, name in enumerate(keypoint_names) 
                         if 'right' in name.lower() and any(kw in name.lower() for kw in hand_keywords)]
    
    if left_hand_indices:
        groups['left_hand'] = (min(left_hand_indices), max(left_hand_indices))
    if right_hand_indices:
        groups['right_hand'] = (min(right_hand_indices), max(right_hand_indices))
    
    return groups, keypoint_names


if __name__ == '__main__':
    # 测试代码
    print("扩展COCO数据集加载器测试")
    
    # 创建模拟配置
    class Config:
        class MODEL:
            IMAGE_SIZE = [256, 256]
            HEATMAP_SIZE = [64, 64]
            SIGMA = 2
    
    config = Config()
    
    # 测试检测分组
    print("\n测试关键点分组检测:")
    # groups, names = get_keypoint_group_names('your_dataset.json')
    # print(f"检测到的分组: {groups}")
