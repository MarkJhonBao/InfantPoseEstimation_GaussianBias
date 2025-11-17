import numpy as np
import torch
import cv2
from pose_transforms import (
    LoadImage, GetBBoxCenterScale, RandomFlip, RandomHalfBody,
    RandomBBoxTransform, TopdownAffine, GenerateTarget, PackPoseInputs,
    build_train_pipeline, build_val_pipeline
)


def create_dummy_image(height=480, width=640):
    """创建一个测试用的虚拟图像"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return img


def create_dummy_keypoints(num_keypoints=17, img_width=640, img_height=480):
    """创建测试用的关键点"""
    # 在图像中心区域生成随机关键点
    center_x, center_y = img_width // 2, img_height // 2
    keypoints = np.random.randn(num_keypoints, 2) * 50 + [center_x, center_y]
    keypoints = np.clip(keypoints, [0, 0], [img_width, img_height])
    keypoints_visible = np.ones(num_keypoints)
    return keypoints, keypoints_visible


def test_load_image():
    """测试LoadImage"""
    print("=" * 50)
    print("测试 LoadImage")
    print("=" * 50)
    
    # 创建临时图像
    dummy_img = create_dummy_image()
    cv2.imwrite('/tmp/test_image.jpg', cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
    
    results = {'img_path': '/tmp/test_image.jpg'}
    transform = LoadImage()
    results = transform(results)
    
    print(f"✓ 图像加载成功")
    print(f"  - 图像形状: {results['img'].shape}")
    print(f"  - 图像类型: {results['img'].dtype}")
    print(f"  - 原始形状: {results['ori_shape']}")
    print()


def test_get_bbox_center_scale():
    """测试GetBBoxCenterScale"""
    print("=" * 50)
    print("测试 GetBBoxCenterScale")
    print("=" * 50)
    
    # 测试 [x, y, w, h] 格式
    results = {'bbox': [100, 150, 200, 300]}
    transform = GetBBoxCenterScale(padding=1.25)
    results = transform(results)
    
    print(f"✓ BBox处理成功")
    print(f"  - 原始bbox: [100, 150, 200, 300]")
    print(f"  - 中心点: {results['center']}")
    print(f"  - 尺度: {results['scale']}")
    print()


def test_random_flip():
    """测试RandomFlip"""
    print("=" * 50)
    print("测试 RandomFlip")
    print("=" * 50)
    
    img = create_dummy_image()
    keypoints, keypoints_visible = create_dummy_keypoints()
    
    # COCO 左右对称关键点对
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), 
                  (11, 12), (13, 14), (15, 16)]
    
    results = {
        'img': img.copy(),
        'center': np.array([320.0, 240.0]),
        'keypoints': keypoints.copy(),
        'keypoints_visible': keypoints_visible,
        'flip_pairs': flip_pairs
    }
    
    transform = RandomFlip(direction='horizontal', prob=1.0)  # 确保翻转
    results_flipped = transform(results)
    
    print(f"✓ 随机翻转成功")
    print(f"  - 原始中心点: [320.0, 240.0]")
    print(f"  - 翻转后中心点: {results_flipped['center']}")
    print(f"  - 翻转标记: {results_flipped.get('flipped', False)}")
    print(f"  - 关键点形状: {results_flipped['keypoints'].shape}")
    print()


def test_random_half_body():
    """测试RandomHalfBody"""
    print("=" * 50)
    print("测试 RandomHalfBody")
    print("=" * 50)
    
    img = create_dummy_image()
    keypoints, keypoints_visible = create_dummy_keypoints()
    
    results = {
        'img': img,
        'center': np.array([320.0, 240.0]),
        'scale': np.array([200.0, 300.0]),
        'keypoints': keypoints,
        'keypoints_visible': keypoints_visible
    }
    
    original_center = results['center'].copy()
    original_scale = results['scale'].copy()
    
    transform = RandomHalfBody(prob=1.0)  # 确保触发
    results = transform(results)
    
    print(f"✓ 随机半身增强成功")
    print(f"  - 原始中心: {original_center}")
    print(f"  - 新中心: {results['center']}")
    print(f"  - 原始尺度: {original_scale}")
    print(f"  - 新尺度: {results['scale']}")
    print()


def test_random_bbox_transform():
    """测试RandomBBoxTransform"""
    print("=" * 50)
    print("测试 RandomBBoxTransform")
    print("=" * 50)
    
    results = {
        'center': np.array([320.0, 240.0]),
        'scale': np.array([200.0, 300.0])
    }
    
    original_center = results['center'].copy()
    original_scale = results['scale'].copy()
    
    transform = RandomBBoxTransform(
        scale_factor=(0.8, 1.2),
        shift_factor=0.1,
        rotate_factor=30,
        prob=1.0
    )
    results = transform(results)
    
    print(f"✓ 随机BBox变换成功")
    print(f"  - 原始中心: {original_center}")
    print(f"  - 变换后中心: {results['center']}")
    print(f"  - 原始尺度: {original_scale}")
    print(f"  - 变换后尺度: {results['scale']}")
    print(f"  - 旋转角度: {results.get('rotation', 0):.2f}°")
    print()


def test_topdown_affine():
    """测试TopdownAffine"""
    print("=" * 50)
    print("测试 TopdownAffine")
    print("=" * 50)
    
    img = create_dummy_image()
    keypoints, _ = create_dummy_keypoints()
    
    results = {
        'img': img,
        'center': np.array([320.0, 240.0]),
        'scale': np.array([250.0, 375.0]),
        'rotation': 15.0,
        'keypoints': keypoints
    }
    
    transform = TopdownAffine(input_size=(192, 256))  # (W, H)
    results = transform(results)
    
    print(f"✓ 仿射变换成功")
    print(f"  - 输入图像形状: {img.shape}")
    print(f"  - 输出图像形状: {results['img'].shape}")
    print(f"  - 仿射矩阵形状: {results['affine_matrix'].shape}")
    print(f"  - 变换后关键点形状: {results['keypoints'].shape}")
    print()


def test_generate_target():
    """测试GenerateTarget"""
    print("=" * 50)
    print("测试 GenerateTarget")
    print("=" * 50)
    
    keypoints = np.array([
        [96, 128],  # 已经在热图坐标系中
        [100, 120],
        [80, 140]
    ], dtype=np.float32)
    
    keypoints_visible = np.array([1, 1, 1])
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    results = {
        'keypoints': keypoints,
        'keypoints_visible': keypoints_visible
    }
    
    transform = GenerateTarget(encoder=codec)
    results = transform(results)
    
    print(f"✓ 生成热图成功")
    print(f"  - 热图形状: {results['heatmaps'].shape}")
    print(f"  - 关键点权重形状: {results['keypoint_weights'].shape}")
    print(f"  - 热图最大值: {results['heatmaps'].max():.4f}")
    print(f"  - 热图最小值: {results['heatmaps'].min():.4f}")
    print()


def test_pack_pose_inputs():
    """测试PackPoseInputs"""
    print("=" * 50)
    print("测试 PackPoseInputs")
    print("=" * 50)
    
    img = create_dummy_image(256, 192)
    heatmaps = np.random.rand(17, 64, 48).astype(np.float32)
    keypoint_weights = np.ones(17, dtype=np.float32)
    
    results = {
        'img': img,
        'heatmaps': heatmaps,
        'keypoint_weights': keypoint_weights,
        'img_path': 'test.jpg',
        'ori_shape': (480, 640),
        'img_shape': (256, 192)
    }
    
    transform = PackPoseInputs()
    results = transform(results)
    
    print(f"✓ 数据打包成功")
    print(f"  - 图像tensor形状: {results['img'].shape}")
    print(f"  - 图像tensor类型: {results['img'].dtype}")
    print(f"  - 热图tensor形状: {results['heatmaps'].shape}")
    print(f"  - 权重tensor形状: {results['keypoint_weights'].shape}")
    print(f"  - 元信息keys: {list(results['data_sample'].keys())}")
    print()


def test_full_pipeline():
    """测试完整pipeline"""
    print("=" * 50)
    print("测试完整训练Pipeline")
    print("=" * 50)
    
    # 创建测试图像
    dummy_img = create_dummy_image()
    cv2.imwrite('/tmp/test_pipeline.jpg', cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
    
    # COCO配置
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), 
                  (11, 12), (13, 14), (15, 16)]
    
    # 准备输入数据
    keypoints, keypoints_visible = create_dummy_keypoints(17)
    
    results = {
        'img_path': '/tmp/test_pipeline.jpg',
        'bbox': [150, 100, 200, 350],
        'keypoints': keypoints,
        'keypoints_visible': keypoints_visible,
        'flip_pairs': flip_pairs
    }
    
    # 构建pipeline
    pipeline = build_train_pipeline(codec, flip_pairs)
    
    # 执行pipeline
    print("执行pipeline步骤:")
    for i, transform in enumerate(pipeline):
        transform_name = transform.__class__.__name__
        print(f"  Step {i+1}: {transform_name}")
        results = transform(results)
    
    print(f"\n✓ 完整pipeline执行成功")
    print(f"  - 输出图像形状: {results['img'].shape}")
    print(f"  - 输出图像范围: [{results['img'].min():.3f}, {results['img'].max():.3f}]")
    
    if 'heatmaps' in results:
        print(f"  - 热图形状: {results['heatmaps'].shape}")
        print(f"  - 热图有效关键点数: {(results['keypoint_weights'] > 0).sum()}")
    
    print()


def test_validation_pipeline():
    """测试验证pipeline"""
    print("=" * 50)
    print("测试验证Pipeline")
    print("=" * 50)
    
    # 创建测试图像
    dummy_img = create_dummy_image()
    cv2.imwrite('/tmp/test_val.jpg', cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    results = {
        'img_path': '/tmp/test_val.jpg',
        'bbox': [150, 100, 200, 350]
    }
    
    # 构建验证pipeline
    pipeline = build_val_pipeline(codec)
    
    print("执行验证pipeline步骤:")
    for i, transform in enumerate(pipeline):
        transform_name = transform.__class__.__name__
        print(f"  Step {i+1}: {transform_name}")
        results = transform(results)
    
    print(f"\n✓ 验证pipeline执行成功")
    print(f"  - 输出图像形状: {results['img'].shape}")
    print()


def visualize_heatmap_example():
    """可视化热图示例"""
    print("=" * 50)
    print("热图可视化示例")
    print("=" * 50)
    
    # 创建简单的关键点
    keypoints = np.array([
        [96, 128],
        [100, 120],
        [80, 140]
    ], dtype=np.float32)
    
    keypoints_visible = np.array([1, 1, 1])
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    results = {
        'keypoints': keypoints,
        'keypoints_visible': keypoints_visible
    }
    
    transform = GenerateTarget(encoder=codec)
    results = transform(results)
    
    heatmaps = results['heatmaps']
    
    print(f"✓ 生成了 {len(keypoints)} 个关键点的热图")
    for i, (kp, hm) in enumerate(zip(keypoints, heatmaps)):
        max_val = hm.max()
        max_pos = np.unravel_index(hm.argmax(), hm.shape)
        print(f"  关键点 {i}: 位置({kp[0]:.1f}, {kp[1]:.1f}) -> "
              f"热图峰值: {max_val:.4f} at {max_pos}")
    print()


def benchmark_transforms():
    """性能基准测试"""
    print("=" * 50)
    print("性能基准测试")
    print("=" * 50)
    
    import time
    
    # 创建测试数据
    dummy_img = create_dummy_image()
    cv2.imwrite('/tmp/benchmark.jpg', cv2.cvtColor(dummy_img, cv2.COLOR_RGB2BGR))
    
    codec = {
        'input_size': (192, 256),
        'heatmap_size': (48, 64),
        'sigma': 2.0
    }
    
    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), 
                  (11, 12), (13, 14), (15, 16)]
    
    keypoints, keypoints_visible = create_dummy_keypoints(17)
    
    # 构建pipeline
    pipeline = build_train_pipeline(codec, flip_pairs)
    
    # 执行多次测试
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        results = {
            'img_path': '/tmp/benchmark.jpg',
            'bbox': [150, 100, 200, 350],
            'keypoints': keypoints.copy(),
            'keypoints_visible': keypoints_visible.copy(),
            'flip_pairs': flip_pairs
        }
        
        for transform in pipeline:
            results = transform(results)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    print(f"✓ 执行 {num_iterations} 次完整pipeline")
    print(f"  - 总时间: {end_time - start_time:.3f}秒")
    print(f"  - 平均时间: {avg_time*1000:.2f}ms/样本")
    print(f"  - 吞吐量: {1/avg_time:.1f}样本/秒")
    print()


if __name__ == '__main__':
    print("\n" + "="*50)
    print("MMPose数据增强Pipeline测试套件")
    print("="*50 + "\n")
    
    # 运行所有测试
    test_load_image()
    test_get_bbox_center_scale()
    test_random_flip()
    test_random_half_body()
    test_random_bbox_transform()
    test_topdown_affine()
    test_generate_target()
    test_pack_pose_inputs()
    test_full_pipeline()
    test_validation_pipeline()
    visualize_heatmap_example()
    benchmark_transforms()
    
    print("="*50)
    print("✓ 所有测试通过!")
    print("="*50)
