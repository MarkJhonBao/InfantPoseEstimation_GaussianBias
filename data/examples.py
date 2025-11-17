import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict


# ============================================================================
# 示例1: 基础使用 - 处理单张图像
# ============================================================================

def example1_basic_usage():
    """基础使用示例"""
    from pose_transforms import (
        LoadImage, GetBBoxCenterScale, RandomFlip,
        TopdownAffine, GenerateTarget, PackPoseInputs
    )
    
    # 配置编码器
    codec = {
        'input_size': (192, 256),  # (W, H)
        'heatmap_size': (48, 64),  # (W, H)
        'sigma': 2.0
    }
    
    # 构建简单pipeline
    pipeline = [
        LoadImage(),
        GetBBoxCenterScale(),
        RandomFlip(direction='horizontal', prob=0.5),
        TopdownAffine(input_size=codec['input_size']),
        GenerateTarget(encoder=codec),
        PackPoseInputs()
    ]
    
    # 准备数据
    results = {
        'img_path': 'data/images/person_001.jpg',
        'bbox': [100, 150, 200, 350],  # [x, y, w, h]
        'keypoints': np.array([  # 示例：5个关键点
            [150, 200],  # 关键点1
            [180, 195],  # 关键点2
            [170, 250],  # 关键点3
            [160, 300],  # 关键点4
            [190, 310],  # 关键点5
        ], dtype=np.float32),
        'keypoints_visible': np.array([1, 1, 1, 1, 0])  # 最后一个不可见
    }
    
    # 执行pipeline
    for transform in pipeline:
        results = transform(results)
    
    # 获取结果
    img = results['img']  # torch.Tensor, shape: (3, 256, 192)
    heatmaps = results['heatmaps']  # torch.Tensor, shape: (5, 64, 48)
    
    print(f"处理完成!")
    print(f"图像shape: {img.shape}")
    print(f"热图shape: {heatmaps.shape}")
    
    return results


# ============================================================================
# 示例2: 使用COCO格式数据
# ============================================================================

def example2_coco_format():
    """COCO格式数据处理示例"""
    from pose_transforms import build_train_pipeline
    
    # COCO配置
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    # COCO 17个关键点的左右对称对
    flip_pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    
    # 构建COCO训练pipeline
    pipeline = build_train_pipeline(codec, flip_pairs)
    
    # COCO格式数据（从标注文件中读取）
    annotation = {
        'image_id': 123,
        'category_id': 1,
        'keypoints': [  # COCO格式: [x1, y1, v1, x2, y2, v2, ...]
            320, 240, 2,  # nose
            310, 230, 2,  # left_eye
            330, 230, 2,  # right_eye
            # ... 共17个关键点
        ],
        'bbox': [250, 150, 150, 300],  # [x, y, w, h]
    }
    
    # 转换为pipeline输入格式
    keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
    results = {
        'img_path': f'data/coco/images/{annotation["image_id"]:012d}.jpg',
        'bbox': annotation['bbox'],
        'keypoints': keypoints[:, :2],  # (17, 2)
        'keypoints_visible': keypoints[:, 2],  # (17,)
        'flip_pairs': flip_pairs
    }
    
    # 执行pipeline
    for transform in pipeline:
        results = transform(results)
    
    return results


# ============================================================================
# 示例3: 自定义Dataset类
# ============================================================================

class COCOKeypointDataset(Dataset):
    """COCO关键点检测数据集"""
    
    def __init__(self, 
                 img_dir: str,
                 ann_file: str,
                 pipeline: List,
                 flip_pairs: List = None):
        self.img_dir = Path(img_dir)
        self.pipeline = pipeline
        self.flip_pairs = flip_pairs or []
        
        # 加载标注
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.images[ann['image_id']]
        
        # 转换COCO关键点格式
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        
        # 准备pipeline输入
        results = {
            'img_path': str(self.img_dir / img_info['file_name']),
            'bbox': ann['bbox'],
            'keypoints': keypoints[:, :2].astype(np.float32),
            'keypoints_visible': keypoints[:, 2].astype(np.float32),
            'flip_pairs': self.flip_pairs,
            'img_id': ann['image_id'],
            'ann_id': ann['id']
        }
        
        # 执行数据增强
        for transform in self.pipeline:
            results = transform(results)
        
        return results


def example3_custom_dataset():
    """自定义数据集示例"""
    from pose_transforms import build_train_pipeline
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8),
                  (9, 10), (11, 12), (13, 14), (15, 16)]
    
    # 构建训练pipeline
    train_pipeline = build_train_pipeline(codec, flip_pairs)
    
    # 创建数据集
    dataset = COCOKeypointDataset(
        img_dir='data/coco/train2017',
        ann_file='data/coco/annotations/person_keypoints_train2017.json',
        pipeline=train_pipeline,
        flip_pairs=flip_pairs
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn  # 需要自定义collate函数
    )
    
    return dataset, dataloader


def custom_collate_fn(batch):
    """自定义collate函数来处理batch数据"""
    imgs = torch.stack([item['img'] for item in batch])
    heatmaps = torch.stack([item['heatmaps'] for item in batch])
    keypoint_weights = torch.stack([item['keypoint_weights'] for item in batch])
    
    # 收集元信息
    data_samples = [item['data_sample'] for item in batch]
    
    return {
        'img': imgs,
        'heatmaps': heatmaps,
        'keypoint_weights': keypoint_weights,
        'data_samples': data_samples
    }


# ============================================================================
# 示例4: 在训练循环中使用
# ============================================================================

def example4_training_loop():
    """训练循环示例"""
    import torch.nn as nn
    import torch.optim as optim
    from pose_transforms import build_train_pipeline
    
    # 假设我们有一个姿态估计模型
    class PoseModel(nn.Module):
        def __init__(self, num_keypoints=17):
            super().__init__()
            # 简化的模型结构
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # ... 更多层
            )
            self.head = nn.Conv2d(64, num_keypoints, 1)
        
        def forward(self, x):
            x = self.backbone(x)
            heatmaps = self.head(x)
            return heatmaps
    
    # 初始化模型
    model = PoseModel(num_keypoints=17)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 配置
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8),
                  (9, 10), (11, 12), (13, 14), (15, 16)]
    
    # 构建pipeline和数据集
    train_pipeline = build_train_pipeline(codec, flip_pairs)
    # dataset = COCOKeypointDataset(...)  # 使用上面定义的数据集
    # dataloader = DataLoader(dataset, ...)
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # for batch in dataloader:
        #     imgs = batch['img']  # (B, 3, H, W)
        #     target_heatmaps = batch['heatmaps']  # (B, K, H', W')
        #     keypoint_weights = batch['keypoint_weights']  # (B, K)
        #     
        #     # 前向传播
        #     pred_heatmaps = model(imgs)
        #     
        #     # 计算损失（加权MSE）
        #     loss = criterion(pred_heatmaps * keypoint_weights.unsqueeze(-1).unsqueeze(-1),
        #                     target_heatmaps * keypoint_weights.unsqueeze(-1).unsqueeze(-1))
        #     
        #     # 反向传播
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     
        #     total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


# ============================================================================
# 示例5: 验证/推理pipeline
# ============================================================================

def example5_inference():
    """推理示例"""
    from pose_transforms import build_val_pipeline
    import torch.nn.functional as F
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    # 构建验证pipeline（无数据增强）
    val_pipeline = build_val_pipeline(codec)
    
    # 准备推理数据
    results = {
        'img_path': 'test_image.jpg',
        'bbox': [100, 100, 200, 300]
    }
    
    # 执行pipeline
    for transform in val_pipeline:
        results = transform(results)
    
    img = results['img'].unsqueeze(0)  # (1, 3, H, W)
    
    # 加载模型并推理
    # model = load_pretrained_model('checkpoint.pth')
    # model.eval()
    # 
    # with torch.no_grad():
    #     pred_heatmaps = model(img)  # (1, K, H', W')
    # 
    # # 从热图中提取关键点坐标
    # keypoints = heatmap_to_keypoints(pred_heatmaps[0])
    
    return results


def heatmap_to_keypoints(heatmaps):
    """从热图中提取关键点坐标"""
    # heatmaps: (K, H, W)
    K, H, W = heatmaps.shape
    
    # 找到每个热图的最大值位置
    heatmaps_flat = heatmaps.view(K, -1)
    max_vals, max_indices = torch.max(heatmaps_flat, dim=1)
    
    y_coords = max_indices // W
    x_coords = max_indices % W
    
    keypoints = torch.stack([x_coords, y_coords], dim=1).float()
    
    # 亚像素精度调整（可选）
    # ... 使用高斯拟合等方法
    
    return keypoints, max_vals


# ============================================================================
# 示例6: 自定义数据增强
# ============================================================================

class CustomColorJitter:
    """自定义颜色抖动增强"""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.prob = prob
    
    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        
        img = results['img'].astype(np.float32) / 255.0
        
        # 亮度调整
        brightness_factor = 1 + np.random.uniform(-self.brightness, self.brightness)
        img = img * brightness_factor
        
        # 对比度调整
        contrast_factor = 1 + np.random.uniform(-self.contrast, self.contrast)
        mean = img.mean()
        img = (img - mean) * contrast_factor + mean
        
        # 饱和度调整（简化版）
        saturation_factor = 1 + np.random.uniform(-self.saturation, self.saturation)
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * saturation_factor
        
        # 裁剪到[0, 1]范围
        img = np.clip(img, 0, 1)
        
        results['img'] = (img * 255).astype(np.uint8)
        
        return results


def example6_custom_augmentation():
    """自定义增强示例"""
    from pose_transforms import (
        LoadImage, GetBBoxCenterScale, RandomFlip,
        TopdownAffine, GenerateTarget, PackPoseInputs
    )
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    # 包含自定义增强的pipeline
    custom_pipeline = [
        LoadImage(),
        GetBBoxCenterScale(),
        RandomFlip(direction='horizontal', prob=0.5),
        CustomColorJitter(prob=0.5),  # 自定义增强
        TopdownAffine(input_size=codec['input_size']),
        GenerateTarget(encoder=codec),
        PackPoseInputs()
    ]
    
    return custom_pipeline


# ============================================================================
# 示例7: 多尺度训练
# ============================================================================

def example7_multi_scale_training():
    """多尺度训练示例"""
    from pose_transforms import (
        LoadImage, GetBBoxCenterScale, RandomFlip,
        TopdownAffine, GenerateTarget, PackPoseInputs
    )
    
    # 不同的输入尺度
    scales = [
        (128, 192),  # 小
        (192, 256),  # 中
        (256, 320),  # 大
    ]
    
    pipelines = []
    for input_size in scales:
        codec = {
            'input_size': input_size,
            'heatmap_size': (input_size[0]//4, input_size[1]//4),
            'sigma': 2.0
        }
        
        pipeline = [
            LoadImage(),
            GetBBoxCenterScale(),
            RandomFlip(direction='horizontal', prob=0.5),
            TopdownAffine(input_size=codec['input_size']),
            GenerateTarget(encoder=codec),
            PackPoseInputs()
        ]
        pipelines.append(pipeline)
    
    # 训练时随机选择一个尺度
    # current_pipeline = random.choice(pipelines)
    
    return pipelines


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    print("MMPose数据增强Pipeline - 使用示例")
    print("="*60)
    
    print("\n示例1: 基础使用")
    print("-"*60)
    print("展示了如何处理单张图像的基本流程")
    
    print("\n示例2: COCO格式数据")
    print("-"*60)
    print("展示了如何处理COCO格式的关键点数据")
    
    print("\n示例3: 自定义Dataset类")
    print("-"*60)
    print("展示了如何创建PyTorch Dataset和DataLoader")
    
    print("\n示例4: 训练循环")
    print("-"*60)
    print("展示了如何在训练循环中使用pipeline")
    
    print("\n示例5: 验证/推理")
    print("-"*60)
    print("展示了如何进行模型推理和关键点提取")
    
    print("\n示例6: 自定义数据增强")
    print("-"*60)
    print("展示了如何添加自定义的数据增强操作")
    
    print("\n示例7: 多尺度训练")
    print("-"*60)
    print("展示了如何进行多尺度训练")
    
    print("\n"+"="*60)
    print("查看代码了解详细实现!")
