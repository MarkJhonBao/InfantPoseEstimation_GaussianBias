
## 🚀 快速开始

### 1. 最简单的使用

```python
from pose_transforms import build_train_pipeline

codec = {
    'input_size': (192, 256),
    'heatmap_size': (48, 64),
    'sigma': 2.0
}

pipeline = build_train_pipeline(codec)

results = {
    'img_path': 'image.jpg',
    'bbox': [100, 100, 200, 300],
    'keypoints': keypoints,  # (N, 2)
    'keypoints_visible': visible  # (N,)
}

for transform in pipeline:
    results = transform(results)
```

## 📊 数据格式速查

### 输入格式
- `img_path`: 图像路径 (str)
- `bbox`: [x, y, w, h] (list/array)
- `keypoints`: (N, 2) numpy array
- `keypoints_visible`: (N,) numpy array, 值为0或1

### 输出格式
- `img`: (3, H, W) torch.Tensor, 归一化到[0,1]
- `heatmaps`: (N, H', W') torch.Tensor
- `keypoint_weights`: (N,) torch.Tensor

## 🎯 常用配置

### COCO 17关键点

```python
flip_pairs = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16)
]

upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lower_body_ids = [11, 12, 13, 14, 15, 16]
```

### 标准输入尺寸

| 模型类型 | input_size | heatmap_size | sigma |
|---------|------------|--------------|-------|
| 轻量级   | (128, 192) | (32, 48)     | 2.0   |
| 标准     | (192, 256) | (48, 64)     | 2.0   |
| 高精度   | (256, 320) | (64, 80)     | 2.0   |

## 🔧 Transform参数

### RandomFlip
```python
RandomFlip(
    direction='horizontal',  # 'horizontal' 或 'vertical'
    prob=0.5                # 概率
)
```

### RandomHalfBody
```python
RandomHalfBody(
    min_total_keypoints=8,
    min_half_keypoints=2,
    prob=0.3
)
```

### RandomBBoxTransform
```python
RandomBBoxTransform(
    scale_factor=(0.75, 1.5),
    shift_factor=0.16,
    rotate_factor=40,
    prob=1.0
)
```

### TopdownAffine
```python
TopdownAffine(
    input_size=(192, 256)  # (W, H)
)
```

### GenerateTarget
```python
codec = {
    'input_size': (192, 256),
    'heatmap_size': (48, 64),
    'sigma': 2.0
}
GenerateTarget(encoder=codec)
```

## 💡 常见模式

### 训练Pipeline
```python
pipeline = [
    LoadImage(),
    GetBBoxCenterScale(),
    RandomFlip(prob=0.5),
    RandomHalfBody(prob=0.3),
    RandomBBoxTransform(),
    TopdownAffine(input_size),
    GenerateTarget(codec),
    PackPoseInputs()
]
```

### 验证Pipeline
```python
pipeline = [
    LoadImage(),
    GetBBoxCenterScale(),
    TopdownAffine(input_size),
    PackPoseInputs()
]
```

### 推理Pipeline（无关键点）
```python
pipeline = [
    LoadImage(),
    GetBBoxCenterScale(),
    TopdownAffine(input_size),
    PackPoseInputs()
]
```

## 🐛 故障排除

### 问题1: 热图全是0
**原因**: 关键点坐标超出图像范围或keypoints_visible设置错误
**解决**: 检查关键点坐标和可见性标记

### 问题2: 翻转后关键点错位
**原因**: 未提供flip_pairs或配置错误
**解决**: 确保flip_pairs包含所有对称关键点对

### 问题3: 仿射变换后图像变形
**原因**: center和scale设置不正确
**解决**: 检查bbox是否正确，确保使用GetBBoxCenterScale

### 问题4: 内存占用高
**原因**: batch_size过大或热图分辨率过高
**解决**: 减小batch_size或降低heatmap_size

## 📝 代码片段

### 创建Dataset
```python
class MyDataset(Dataset):
    def __init__(self, data_list, pipeline):
        self.data_list = data_list
        self.pipeline = pipeline
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        results = {
            'img_path': data['img_path'],
            'bbox': data['bbox'],
            'keypoints': data['keypoints'],
            'keypoints_visible': data['keypoints_visible']
        }
        for transform in self.pipeline:
            results = transform(results)
        return results
```

### 自定义Transform
```python
class MyTransform:
    def __init__(self, param1):
        self.param1 = param1
    
    def __call__(self, results):
        # 处理results
        return results
```

### 训练循环
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        imgs = batch['img']
        heatmaps = batch['heatmaps']
        
        pred = model(imgs)
        loss = criterion(pred, heatmaps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🔗 相关资源

- GitHub: [项目地址]
- 文档: README.md
- 示例: examples.py
- 测试: test_transforms.py

