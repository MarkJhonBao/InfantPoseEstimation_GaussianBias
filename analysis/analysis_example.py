"""
完整的神经网络定量分析示例
针对早产儿姿态估计模型

使用方法：
python run_quantitative_analysis.py --checkpoint model.pth --data_dir ./data
"""

import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.pose_hrnet import PoseHighResolutionNet
from data.coco_dataset import PreemieCocoDataset
from config import get_config
from utils.postprocess import fused_decode
from neural_network_analysis import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./analysis_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=100)
    return parser.parse_args()


def collect_predictions(model, dataloader, device, num_samples=100):
    """收集模型预测结果"""
    model.eval()
    
    predictions = []
    ground_truths = []
    confidences = []
    images_list = []
    pred_heatmaps_list = []
    gt_heatmaps_list = []
    
    print("收集预测结果...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if len(predictions) >= num_samples:
                break
            
            images = batch['image'].to(device)
            gt_coords = batch['target_coords'].cpu().numpy()
            gt_heatmaps = batch['target_heatmap'].cpu().numpy()
            
            # 前向传播
            outputs = model(images)
            
            # 解码预测
            pred_coords, maxvals = fused_decode(
                outputs['heatmaps'],
                outputs.get('coords', None),
                batch.get('center'),
                batch.get('scale')
            )
            
            # 保存结果
            batch_size = images.shape[0]
            for i in range(batch_size):
                if len(predictions) >= num_samples:
                    break
                
                predictions.append(pred_coords[i].cpu().numpy())
                ground_truths.append(gt_coords[i])
                confidences.append(maxvals[i].cpu().numpy())
                images_list.append(images[i].cpu())
                pred_heatmaps_list.append(outputs['heatmaps'][i].cpu().numpy())
                gt_heatmaps_list.append(gt_heatmaps[i])
    
    return {
        'predictions': predictions,
        'ground_truths': ground_truths,
        'confidences': confidences,
        'images': images_list,
        'pred_heatmaps': pred_heatmaps_list,
        'gt_heatmaps': gt_heatmaps_list
    }


def run_analysis(args):
    """运行完整的定量分析"""
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("早产儿姿态估计模型 - 定量分析")
    print("="*80)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', get_config())
    
    model = PoseHighResolutionNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✓ 模型加载完成")
    print(f"   设备: {device}")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    dataset = PreemieCocoDataset(
        config,
        ann_file=os.path.join(args.data_dir, 'annotations/val.json'),
        img_dir=os.path.join(args.data_dir, 'images/val'),
        is_train=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"   ✓ 数据加载完成")
    print(f"   样本数: {len(dataset)}")
    
    # 关键点名称
    joint_names = [
        'nose', 'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip'
    ]
    
    # 3. 收集预测结果
    print("\n3. 收集预测结果...")
    results = collect_predictions(model, dataloader, device, args.num_samples)
    print(f"   ✓ 收集了 {len(results['predictions'])} 个样本")
    
    # ===== 性能分析 =====
    print("\n" + "="*80)
    print("性能指标分析")
    print("="*80)
    
    perf_analyzer = PerformanceAnalyzer()
    
    # 3.1 关键点准确率热图
    print("\n3.1 绘制关键点准确率热图...")
    fig1 = perf_analyzer.plot_keypoint_accuracy_heatmap(
        results['predictions'],
        results['ground_truths'],
        joint_names
    )
    fig1.savefig(f'{args.output_dir}/01_keypoint_accuracy_heatmap.png', 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("   ✓ 保存: 01_keypoint_accuracy_heatmap.png")
    
    # 3.2 误差分布
    print("\n3.2 绘制误差分布图...")
    fig2 = perf_analyzer.plot_error_distribution(
        results['predictions'],
        results['ground_truths'],
        joint_names
    )
    fig2.savefig(f'{args.output_dir}/02_error_distribution.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("   ✓ 保存: 02_error_distribution.png")
    
    # 3.3 置信度vs准确率
    print("\n3.3 分析置信度与准确率关系...")
    fig3 = perf_analyzer.plot_confidence_vs_accuracy(
        results['predictions'],
        results['ground_truths'],
        results['confidences'],
        threshold=10
    )
    fig3.savefig(f'{args.output_dir}/03_confidence_vs_accuracy.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("   ✓ 保存: 03_confidence_vs_accuracy.png")
    
    # 3.4 PR曲线
    print("\n3.4 绘制PR曲线...")
    fig4 = perf_analyzer.plot_pr_curve(
        results['predictions'],
        results['ground_truths'],
        results['confidences'],
        threshold=10
    )
    fig4.savefig(f'{args.output_dir}/04_pr_curve.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("   ✓ 保存: 04_pr_curve.png")
    
    # ===== 特征可视化 =====
    print("\n" + "="*80)
    print("特征可视化分析")
    print("="*80)
    
    feature_viz = FeatureVisualizer()
    
    # 4.1 热图质量对比
    print("\n4.1 可视化热图质量...")
    sample_idx = 0
    fig5 = feature_viz.visualize_heatmap_quality(
        results['pred_heatmaps'][sample_idx],
        results['gt_heatmaps'][sample_idx],
        joint_names
    )
    fig5.savefig(f'{args.output_dir}/05_heatmap_quality.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print("   ✓ 保存: 05_heatmap_quality.png")
    
    # 4.2 特征图可视化
    print("\n4.2 可视化卷积特征图...")
    analyzer = NeuralNetworkAnalyzer(model, device)
    
    # 获取一个样本
    sample_image = results['images'][0].unsqueeze(0).to(device)
    _ = model(sample_image)
    
    # 可视化不同层的特征
    layer_names = ['layer1', 'stage2', 'stage3']
    for idx, layer_name in enumerate(layer_names):
        if layer_name in analyzer.features:
            fig = feature_viz.visualize_feature_maps(
                analyzer.features[layer_name],
                layer_name,
                num_samples=16
            )
            if fig:
                fig.savefig(f'{args.output_dir}/06_{idx}_features_{layer_name}.png',
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"   ✓ 保存: 06_{idx}_features_{layer_name}.png")
    
    # ===== Grad-CAM可视化 =====
    print("\n" + "="*80)
    print("Grad-CAM注意力可视化")
    print("="*80)
    
    print("\n5. 生成Grad-CAM...")
    # 找到最后一个卷积层
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = name
    
    if target_layer:
        gradcam = GradCAMVisualizer(model, target_layer)
        
        for i in range(min(3, len(results['images']))):
            sample = results['images'][i].unsqueeze(0).to(device)
            cam = gradcam.generate_cam(sample)
            
            fig = gradcam.visualize_gradcam(sample, cam)
            fig.savefig(f'{args.output_dir}/07_{i}_gradcam.png',
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"   ✓ 保存了3个Grad-CAM样本")
    
    # ===== 模型复杂度分析 =====
    print("\n" + "="*80)
    print("模型复杂度分析")
    print("="*80)
    
    complexity_analyzer = ModelComplexityAnalyzer()
    
    # 6.1 参数统计
    print("\n6.1 统计模型参数...")
    params = complexity_analyzer.count_parameters(model)
    print(f"   总参数: {params['total']:,}")
    print(f"   可训练参数: {params['trainable']:,}")
    print(f"   不可训练参数: {params['non_trainable']:,}")
    
    # 6.2 参数分布
    print("\n6.2 分析参数分布...")
    fig8 = complexity_analyzer.analyze_layer_parameters(model)
    fig8.savefig(f'{args.output_dir}/08_parameter_distribution.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print("   ✓ 保存: 08_parameter_distribution.png")
    
    # 6.3 推理时间
    print("\n6.3 测量推理时间...")
    fig9, time_stats = complexity_analyzer.measure_inference_time(
        model,
        input_size=(1, 3, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]),
        num_runs=100
    )
    fig9.savefig(f'{args.output_dir}/09_inference_time.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig9)
    
    print(f"   平均推理时间: {time_stats['mean']:.2f} ms")
    print(f"   标准差: {time_stats['std']:.2f} ms")
    print(f"   最小/最大: {time_stats['min']:.2f} / {time_stats['max']:.2f} ms")
    print("   ✓ 保存: 09_inference_time.png")
    
    # ===== 生成分析报告 =====
    print("\n" + "="*80)
    print("生成分析报告")
    print("="*80)
    
    report_path = os.path.join(args.output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("早产儿姿态估计模型 - 定量分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 模型信息\n")
        f.write("-"*80 + "\n")
        f.write(f"模型: {config.MODEL.NAME}\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"关键点数量: {config.MODEL.NUM_JOINTS}\n")
        f.write(f"输入尺寸: {config.MODEL.IMAGE_SIZE}\n\n")
        
        f.write("2. 模型复杂度\n")
        f.write("-"*80 + "\n")
        f.write(f"总参数: {params['total']:,}\n")
        f.write(f"可训练参数: {params['trainable']:,}\n")
        f.write(f"平均推理时间: {time_stats['mean']:.2f} ms\n")
        f.write(f"FPS: {1000/time_stats['mean']:.1f}\n\n")
        
        f.write("3. 性能指标\n")
        f.write("-"*80 + "\n")
        
        # 计算平均误差
        all_errors = []
        for pred, gt in zip(results['predictions'], results['ground_truths']):
            for j in range(len(pred)):
                error = np.linalg.norm(pred[j] - gt[j])
                all_errors.append(error)
        
        f.write(f"平均定位误差: {np.mean(all_errors):.2f} pixels\n")
        f.write(f"误差标准差: {np.std(all_errors):.2f} pixels\n")
        f.write(f"中位数误差: {np.median(all_errors):.2f} pixels\n\n")
        
        # 各关键点误差
        f.write("4. 各关键点平均误差\n")
        f.write("-"*80 + "\n")
        for j, name in enumerate(joint_names):
            joint_errors = []
            for pred, gt in zip(results['predictions'], results['ground_truths']):
                error = np.linalg.norm(pred[j] - gt[j])
                joint_errors.append(error)
            f.write(f"{name:20s}: {np.mean(joint_errors):6.2f} ± {np.std(joint_errors):5.2f} pixels\n")
        
        f.write("\n5. 生成的可视化\n")
        f.write("-"*80 + "\n")
        f.write("01_keypoint_accuracy_heatmap.png  - 关键点准确率热图\n")
        f.write("02_error_distribution.png         - 误差分布图\n")
        f.write("03_confidence_vs_accuracy.png     - 置信度vs准确率\n")
        f.write("04_pr_curve.png                   - PR曲线\n")
        f.write("05_heatmap_quality.png            - 热图质量对比\n")
        f.write("06_x_features_xxx.png             - 特征图可视化\n")
        f.write("07_x_gradcam.png                  - Grad-CAM可视化\n")
        f.write("08_parameter_distribution.png     - 参数分布\n")
        f.write("09_inference_time.png             - 推理时间分析\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"   ✓ 报告保存: {report_path}")
    
    # ===== 完成 =====
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有结果已保存到: {args.output_dir}")
    print("\n生成的文件:")
    for file in sorted(os.listdir(args.output_dir)):
        print(f"  - {file}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    print("\n早产儿姿态估计模型 - 定量分析工具")
    print("="*80)
    print(f"检查点: {args.checkpoint}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"样本数量: {args.num_samples}")
    print("="*80 + "\n")
    
    results = run_analysis(args)
    
    print("\n✓ 全部分析完成！")
