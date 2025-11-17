# 数据增强 Pipeline - PyTorch实现


## 📋 功能特性

### 已实现的Transform

1. **LoadImage** - 图像加载
   - 支持从文件路径加载图像
   - 自动BGR到RGB转换
   - 支持float32转换

2. **GetBBoxCenterScale** - 边界框处理
   - 从bbox计算中心点和尺度
   - 支持[x, y, w, h]和[x1, y1, x2, y2]格式
   - 可配置padding系数

3. **RandomFlip** - 随机翻转
   - 支持水平/垂直翻转
   - 自动处理关键点坐标翻转
   - 支持左右对称关键点交换（通过flip_pairs）

4. **RandomHalfBody** - 随机半身增强
   - 随机选择上半身或下半身
   - 自动调整bbox适应选中区域
   - 可配置最小关键点数量

5. **RandomBBoxTransform** - 随机边界框变换
   - 随机缩放（scale）
   - 随机平移（shift）
   - 随机旋转（rotation）

6. **TopdownAffine** - 仿射变换
   - 将图像和关键点变换到固定尺寸
   - 支持旋转、缩放、平移
   - 使用仿射变换矩阵

7. **GenerateTarget** - 生成训练目标
   - 生成高斯热图（Gaussian Heatmap）
   - 可配置热图尺寸和sigma
   - 自动处理关键点可见性

8. **PackPoseInputs** - 打包输入
   - 转换为PyTorch Tensor
   - 图像归一化到[0, 1]
   - 打包元信息

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy opencv-python pillow
```

### 基本使用

```python
from pose_transforms import build_train_pipeline, build_val_pipeline
import numpy as np

# 配置编码器
codec = {
    'input_size': (192, 256),  # (W, H)
    'heatmap_size': (48, 64),  # (W, H)
    'sigma': 2.0
}

# COCO 左右对称关键点对
flip_pairs = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16)
]

# 构建训练pipeline
train_pipeline = build_train_pipeline(codec, flip_pairs)

# 准备输入数据
results = {
    'img_path': 'path/to/image.jpg',
    'bbox': [100, 100, 200, 300],  # [x, y, w, h]
    'keypoints': np.random.rand(17, 2) * 100 + 100,  # 17个关键点
    'keypoints_visible': np.ones(17),
    'flip_pairs': flip_pairs
}

# 执行pipeline
for transform in train_pipeline:
    results = transform(results)

# 获取结果
img_tensor = results['img']  # (3, H, W)
heatmaps = results['heatmaps']  # (num_keypoints, heatmap_h, heatmap_w)
```

### 自定义Pipeline

```python
from pose_transforms import (
    LoadImage, GetBBoxCenterScale, RandomFlip,
    TopdownAffine, GenerateTarget, PackPoseInputs
)

# 自定义pipeline
custom_pipeline = [
    LoadImage(to_float32=False),
    GetBBoxCenterScale(padding=1.5),
    RandomFlip(direction='horizontal', prob=0.5),
    TopdownAffine(input_size=(256, 256)),
    GenerateTarget(encoder=codec),
    PackPoseInputs()
]

# 使用自定义pipeline
results = {'img_path': 'image.jpg', 'bbox': [0, 0, 100, 100]}
for transform in custom_pipeline:
    results = transform(results)
```

## 📊 数据格式

### 输入格式

```python
{
    'img_path': str,                    # 图像路径
    'bbox': [x, y, w, h],              # 边界框 [x, y, width, height]
    'keypoints': np.ndarray,            # (num_keypoints, 2 or 3)
    'keypoints_visible': np.ndarray,    # (num_keypoints,) 0或1
    'flip_pairs': List[Tuple[int, int]] # 左右对称关键点对
}
```

### 输出格式

```python
{
    'img': torch.Tensor,                # (3, H, W) 归一化到[0, 1]
    'heatmaps': torch.Tensor,           # (num_keypoints, heatmap_h, heatmap_w)
    'keypoint_weights': torch.Tensor,   # (num_keypoints,)
    'keypoints': torch.Tensor,          # (num_keypoints, 2 or 3)
    'data_sample': Dict                 # 元信息
}
```

## 🎯 COCO格式示例

COCO数据集有17个关键点，索引如下：

```python
# COCO 17个关键点
keypoint_names = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 左右对称关键点对
flip_pairs = [
    (1, 2),   # 左眼 <-> 右眼
    (3, 4),   # 左耳 <-> 右耳
    (5, 6),   # 左肩 <-> 右肩
    (7, 8),   # 左肘 <-> 右肘
    (9, 10),  # 左腕 <-> 右腕
    (11, 12), # 左臀 <-> 右臀
    (13, 14), # 左膝 <-> 右膝
    (15, 16)  # 左踝 <-> 右踝
]

# 上半身关键点索引
upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 下半身关键点索引
lower_body_ids = [11, 12, 13, 14, 15, 16]
```

## 🔧 参数配置

### RandomFlip

```python
RandomFlip(
    direction='horizontal',  # 'horizontal' 或 'vertical'
    prob=0.5                # 翻转概率
)
```

### RandomHalfBody

```python
RandomHalfBody(
    min_total_keypoints=8,    # 最小总关键点数
    min_half_keypoints=2,     # 最小半身关键点数
    prob=0.3,                 # 触发概率
    upper_body_ids=[0,1,...], # 上半身关键点索引
    lower_body_ids=[11,12,...]# 下半身关键点索引
)
```

### RandomBBoxTransform

```python
RandomBBoxTransform(
    scale_factor=(0.75, 1.5), # 缩放范围
    shift_factor=0.16,         # 平移因子
    rotate_factor=40,          # 旋转角度范围 [-40, 40]
    prob=1.0                   # 触发概率
)
```

### TopdownAffine

```python
TopdownAffine(
    input_size=(192, 256)  # 输出尺寸 (W, H)
)
```

### GenerateTarget

```python
codec = {
    'input_size': (192, 256),   # 输入图像尺寸 (W, H)
    'heatmap_size': (48, 64),   # 热图尺寸 (W, H)
    'sigma': 2.0                # 高斯核标准差
}
GenerateTarget(encoder=codec)
```

## 🧪 测试

运行完整测试套件：

```bash
python test_transforms.py
```

测试包括：
- ✅ 单个transform功能测试
- ✅ 完整pipeline测试
- ✅ 训练和验证pipeline测试
- ✅ 热图生成可视化
- ✅ 性能基准测试

## 📈 性能

在标准配置下（input_size=192x256, 17个关键点）：
- 单个样本处理时间: ~10-20ms
- 吞吐量: ~50-100 samples/sec（单线程CPU）

## 🔍 注意事项

1. **坐标系统**
   - 所有坐标使用(x, y)格式
   - 图像尺寸使用(H, W)格式
   - 输入输出尺寸使用(W, H)格式

2. **关键点可见性**
   - 0: 不可见
   - 1: 被遮挡但标注
   - 2: 可见

3. **BBox格式**
   - 支持[x, y, w, h]格式
   - 支持[x1, y1, x2, y2]格式（自动检测）

4. **热图生成**
   - 使用高斯分布
   - 自动处理边界情况
   - 支持关键点权重

| 特性 | 本实现 | 
|------|--------|
| 依赖 | torch, numpy, cv2 |
| 配置 | Python字典 | 
| 扩展性 | 简单直接 | 
| 性能 | 相近 | 

## 📝 示例项目结构

```
project/
├── pose_transforms.py      # 核心transforms实现
├── test_transforms.py      # 测试文件
├── README.md              # 本文档
└── your_training.py       # 你的训练代码
```

## 💡 扩展建议

1. **添加新的数据增强**
   ```python
   class CustomTransform:
       def __init__(self, param1, param2):
           self.param1 = param1
           self.param2 = param2
       
       def __call__(self, results: Dict) -> Dict:
           # 实现你的增强逻辑
           return results
   ```

2. **集成到训练循环**
   ```python
   from torch.utils.data import Dataset, DataLoader
   
   class PoseDataset(Dataset):
       def __init__(self, data_list, pipeline):
           self.data_list = data_list
           self.pipeline = pipeline
       
       def __getitem__(self, idx):
           data_info = self.data_list[idx]
           results = {'img_path': data_info['img_path'], ...}
           
           for transform in self.pipeline:
               results = transform(results)
           
           return results['img'], results['heatmaps']
   ```

## 🐛 常见问题

**Q: 为什么热图全是0?**
A: 检查关键点坐标是否在图像范围内，以及keypoints_visible是否正确设置。

**Q: 翻转后关键点位置不对?**
A: 确保提供了正确的flip_pairs参数。

**Q: 内存占用过高?**
A: 减小batch_size或降低热图分辨率。

## 📄 许可

MIT License

## 🙏 致谢

