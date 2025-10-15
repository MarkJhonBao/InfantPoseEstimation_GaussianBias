"""
神经网络定量分析可视化工具
专门针对早产儿姿态估计模型

包含的分析方法：
1. 性能指标可视化
2. 特征图可视化
3. 注意力机制可视化（Grad-CAM）
4. 预测误差分析
5. 模型复杂度分析
6. 训练过程分析
7. 关键点检测质量分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from scipy import stats
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class NeuralNetworkAnalyzer:
    """神经网络定量分析器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 存储中间特征
        self.features = {}
        self.gradients = {}
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # 为关键层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))


class PerformanceAnalyzer:
    """性能指标分析器"""
    
    @staticmethod
    def plot_keypoint_accuracy_heatmap(predictions, ground_truth, joint_names):
        """
        绘制关键点准确率热图
        """
        num_samples = len(predictions)
        num_joints = len(joint_names)
        
        # 计算每个关键点的准确率（PCK）
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        accuracy_matrix = np.zeros((num_joints, len(thresholds)))
        
        for j in range(num_joints):
            for t_idx, threshold in enumerate(thresholds):
                correct = 0
                for pred, gt in zip(predictions, ground_truth):
                    distance = np.linalg.norm(pred[j] - gt[j])
                    bbox_size = np.linalg.norm([gt.max() - gt.min()])
                    if distance < threshold * bbox_size:
                        correct += 1
                accuracy_matrix[j, t_idx] = correct / num_samples
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(accuracy_matrix, 
                   xticklabels=[f'{t:.2f}' for t in thresholds],
                   yticklabels=joint_names,
                   annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Accuracy'},
                   ax=ax)
        ax.set_xlabel('PCK Threshold')
        ax.set_ylabel('Joint')
        ax.set_title('Joint Detection Accuracy Heatmap')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_error_distribution(predictions, ground_truth, joint_names):
        """
        绘制误差分布图
        """
        num_joints = len(joint_names)
        errors = defaultdict(list)
        
        for pred, gt in zip(predictions, ground_truth):
            for j in range(num_joints):
                error = np.linalg.norm(pred[j] - gt[j])
                errors[joint_names[j]].append(error)
        
        # 创建箱线图
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 箱线图
        data_to_plot = [errors[name] for name in joint_names]
        bp = axes[0].boxplot(data_to_plot, labels=joint_names, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        axes[0].set_ylabel('Error (pixels)')
        axes[0].set_title('Keypoint Localization Error Distribution')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 小提琴图
        positions = range(1, num_joints + 1)
        parts = axes[1].violinplot(data_to_plot, positions=positions, 
                                   showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(joint_names, rotation=45, ha='right')
        axes[1].set_ylabel('Error (pixels)')
        axes[1].set_title('Error Distribution (Violin Plot)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confidence_vs_accuracy(predictions, ground_truth, confidences, threshold=10):
        """
        绘制置信度与准确率的关系
        """
        # 计算每个预测的准确性
        is_accurate = []
        conf_scores = []
        
        for pred, gt, conf in zip(predictions, ground_truth, confidences):
            for j in range(len(pred)):
                error = np.linalg.norm(pred[j] - gt[j])
                is_accurate.append(1 if error < threshold else 0)
                conf_scores.append(conf[j])
        
        # 按置信度分箱
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            mask = (np.array(conf_scores) >= bins[i]) & (np.array(conf_scores) < bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(np.array(is_accurate)[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 置信度 vs 准确率
        ax1.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Confidence vs Accuracy (Calibration Curve)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # 置信度分布
        ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pr_curve(predictions, ground_truth, confidences, threshold=10):
        """
        绘制PR曲线（Precision-Recall）
        """
        # 计算不同置信度阈值下的precision和recall
        conf_thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        
        for conf_thresh in conf_thresholds:
            tp = fp = fn = 0
            
            for pred, gt, conf in zip(predictions, ground_truth, confidences):
                for j in range(len(pred)):
                    error = np.linalg.norm(pred[j] - gt[j])
                    is_correct = error < threshold
                    is_detected = conf[j] > conf_thresh
                    
                    if is_detected and is_correct:
                        tp += 1
                    elif is_detected and not is_correct:
                        fp += 1
                    elif not is_detected and is_correct:
                        fn += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP (Average Precision)
        ap = np.trapz(precisions, recalls)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions, 'b-', linewidth=2, label=f'AP = {ap:.3f}')
        ax.fill_between(recalls, precisions, alpha=0.3)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig


class FeatureVisualizer:
    """特征可视化器"""
    
    @staticmethod
    def visualize_feature_maps(features, layer_name, num_samples=16):
        """
        可视化卷积层特征图
        """
        if len(features.shape) != 4:  # [B, C, H, W]
            print(f"Invalid feature shape: {features.shape}")
            return None
        
        batch_size, num_channels, height, width = features.shape
        num_samples = min(num_samples, num_channels)
        
        # 选择要显示的特征图
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i in range(num_samples):
            feature_map = features[0, i].cpu().numpy()
            
            # 归一化到[0, 1]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def visualize_heatmap_quality(pred_heatmap, gt_heatmap, joint_names):
        """
        可视化预测热图与真实热图的对比
        """
        num_joints = pred_heatmap.shape[0]
        
        fig, axes = plt.subplots(num_joints, 3, figsize=(12, 3*num_joints))
        
        for j in range(num_joints):
            # 真实热图
            axes[j, 0].imshow(gt_heatmap[j], cmap='hot')
            axes[j, 0].set_title(f'{joint_names[j]} - Ground Truth')
            axes[j, 0].axis('off')
            
            # 预测热图
            axes[j, 1].imshow(pred_heatmap[j], cmap='hot')
            axes[j, 1].set_title(f'{joint_names[j]} - Prediction')
            axes[j, 1].axis('off')
            
            # 差异图
            diff = np.abs(pred_heatmap[j] - gt_heatmap[j])
            im = axes[j, 2].imshow(diff, cmap='coolwarm')
            axes[j, 2].set_title(f'{joint_names[j]} - Difference')
            axes[j, 2].axis('off')
            plt.colorbar(im, ax=axes[j, 2])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_feature_tsne(features, labels=None):
        """
        使用t-SNE降维可视化特征
        """
        # features: [N, D]
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features.cpu().numpy())
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Feature Space Visualization (t-SNE)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class GradCAMVisualizer:
    """Grad-CAM注意力可视化"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_image, target_class=None):
        """
        生成Grad-CAM热图
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_image)
        
        if isinstance(output, dict):
            output = output['heatmaps']
        
        # 如果未指定目标，使用最大响应
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        output[0, target_class].sum().backward()
        
        # 计算权重
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # 加权组合
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    @staticmethod
    def visualize_gradcam(image, cam, alpha=0.5):
        """
        可视化Grad-CAM
        """
        # 调整CAM大小
        cam_resized = cv2.resize(cam[0, 0].cpu().numpy(), 
                                (image.shape[3], image.shape[2]))
        
        # 转换为热图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 原始图像
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        image_np = np.uint8(255 * image_np)
        
        # 叠加
        overlay = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig


class TrainingAnalyzer:
    """训练过程分析器"""
    
    @staticmethod
    def plot_training_curves(history):
        """
        绘制训练曲线
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        if 'train_acc' in history:
            axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
            axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 损失分解
        if 'heatmap_loss' in history and 'morph_loss' in history:
            axes[1, 1].plot(epochs, history['heatmap_loss'], label='Heatmap Loss', linewidth=2)
            axes[1, 1].plot(epochs, history['morph_loss'], label='Morphology Loss', linewidth=2)
            if 'reg_loss' in history:
                axes[1, 1].plot(epochs, history['reg_loss'], label='Regression Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Loss Components')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_gradient_flow(named_parameters):
        """
        可视化梯度流
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in named_parameters:
            if param.requires_grad and param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color='c', label='Max Gradient')
        ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color='b', label='Mean Gradient')
        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color='k')
        ax.set_xticks(range(0, len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation=90, ha='right')
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom=-0.001)
        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient')
        ax.set_title('Gradient Flow')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig


class ModelComplexityAnalyzer:
    """模型复杂度分析器"""
    
    @staticmethod
    def count_parameters(model):
        """统计模型参数"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def analyze_layer_parameters(model):
        """分析各层参数分布"""
        layer_params = []
        layer_names = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    layer_names.append(name)
                    layer_params.append(num_params)
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 参数分布条形图
        y_pos = np.arange(len(layer_names))
        ax1.barh(y_pos, layer_params, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([name.split('.')[-1] for name in layer_names], fontsize=8)
        ax1.set_xlabel('Number of Parameters')
        ax1.set_title('Parameters per Layer')
        ax1.grid(True, alpha=0.3)
        
        # 参数占比饼图（只显示前10层）
        top_10_idx = np.argsort(layer_params)[-10:]
        top_10_params = [layer_params[i] for i in top_10_idx]
        top_10_names = [layer_names[i].split('.')[-1] for i in top_10_idx]
        other_params = sum(layer_params) - sum(top_10_params)
        
        if other_params > 0:
            top_10_params.append(other_params)
            top_10_names.append('Others')
        
        ax2.pie(top_10_params, labels=top_10_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Parameter Distribution (Top 10 Layers)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def measure_inference_time(model, input_size=(1, 3, 256, 256), num_runs=100):
        """测量推理时间"""
        import time
        
        model.eval()
        device = next(model.parameters()).device
        
        # 预热
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测量
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append((time.time() - start) * 1000)  # ms
        
        # 统计
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 时间分布直方图
        ax1.hist(times, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(stats['mean'], color='r', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}ms")
        ax1.axvline(stats['median'], color='g', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.2f}ms")
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 时间序列
        ax2.plot(times, alpha=0.7)
        ax2.axhline(stats['mean'], color='r', linestyle='--', linewidth=2)
        ax2.fill_between(range(len(times)), 
                         stats['mean'] - stats['std'], 
                         stats['mean'] + stats['std'],
                         alpha=0.3, color='red')
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Series')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, stats


# 使用示例和主函数
def create_comprehensive_analysis_report(model, dataloader, output_dir='./analysis_results'):
    """
    创建完整的分析报告
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("神经网络定量分析报告生成")
    print("="*80)
    
    # 收集预测和真实值
    predictions = []
    ground_truths = []
    confidences = []
    
    model.eval()
    device = next(model.parameters()).device)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            # 假设outputs是字典格式
            pred_heatmaps = outputs['heatmaps']
            # 解码为关键点坐标（简化版）
            # 实际应该使用完整的解码函数
            
            # 这里需要根据实际情况调整
            break
    
    # 1. 性能分析
    print("\n1. 性能指标分析...")
    perf_analyzer = PerformanceAnalyzer()
    
    # 这里需要实际的预测和真实值数据
    # fig1 = perf_analyzer.plot_keypoint_accuracy_heatmap(predictions, ground_truths, joint_names)
    # fig1.savefig(f'{output_dir}/accuracy_heatmap.png', dpi=150, bbox_inches='tight')
    
    # 2. 模型复杂度分析
    print("\n2. 模型复杂度分析...")
    complexity_analyzer = ModelComplexityAnalyzer()
    
    params = complexity_analyzer.count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    fig2 = complexity_analyzer.analyze_layer_parameters(model)
    fig2.savefig(f'{output_dir}/parameter_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    fig3, time_stats = complexity_analyzer.measure_inference_time(model)
    print(f"   Mean inference time: {time_stats['mean']:.2f} ms")
    fig3.savefig(f'{output_dir}/inference_time.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print("\n✓ 分析完成！结果保存在:", output_dir)


if __name__ == '__main__':
    # 这是一个演示
    print("神经网络定量分析可视化工具")
    print("请在实际项目中导入并使用这些类")
