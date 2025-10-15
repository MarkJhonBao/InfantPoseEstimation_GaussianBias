"""
高级神经网络分析工具
包含更深入的网络内部分析方法
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import cv2


class ActivationAnalyzer:
    """激活值分析器"""
    
    @staticmethod
    def analyze_activation_distribution(activations):
        """
        分析激活值分布
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 转换为numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        act_flat = activations.flatten()
        
        # 1. 直方图
        axes[0, 0].hist(act_flat, bins=100, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Activation Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Activation Distribution')
        axes[0, 0].axvline(0, color='r', linestyle='--', label='Zero')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CDF
        sorted_act = np.sort(act_flat)
        cdf = np.arange(len(sorted_act)) / len(sorted_act)
        axes[0, 1].plot(sorted_act, cdf, linewidth=2)
        axes[0, 1].set_xlabel('Activation Value')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 箱线图（按通道）
        if len(activations.shape) == 4:  # [B, C, H, W]
            channel_means = activations.mean(axis=(0, 2, 3))
            axes[1, 0].boxplot([activations[:, i].flatten() for i in range(min(10, activations.shape[1]))],
                              labels=[f'Ch{i}' for i in range(min(10, activations.shape[1]))])
            axes[1, 0].set_xlabel('Channel')
            axes[1, 0].set_ylabel('Activation Value')
            axes[1, 0].set_title('Activation by Channel (First 10)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 统计信息
        stats_text = f"""
        Mean: {act_flat.mean():.4f}
        Std: {act_flat.std():.4f}
        Min: {act_flat.min():.4f}
        Max: {act_flat.max():.4f}
        Median: {np.median(act_flat):.4f}
        
        Dead neurons: {(act_flat == 0).sum() / len(act_flat) * 100:.2f}%
        Negative: {(act_flat < 0).sum() / len(act_flat) * 100:.2f}%
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def analyze_dead_neurons(model, dataloader, device, threshold=1e-6):
        """
        分析死神经元（激活值始终为0的神经元）
        """
        model.eval()
        
        # 收集所有层的激活
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                activations[name].append(output.detach().cpu())
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 前向传播
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                _ = model(images)
                break  # 只用一个batch
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 分析死神经元
        dead_neuron_ratios = {}
        for name, acts in activations.items():
            act_tensor = torch.cat(acts, dim=0)
            # 计算每个神经元的平均激活
            if len(act_tensor.shape) == 4:  # [B, C, H, W]
                neuron_mean = act_tensor.mean(dim=(0, 2, 3))
            elif len(act_tensor.shape) == 2:  # [B, C]
                neuron_mean = act_tensor.mean(dim=0)
            else:
                continue
            
            dead_ratio = (neuron_mean < threshold).float().mean().item()
            dead_neuron_ratios[name] = dead_ratio
        
        # 可视化
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_names = list(dead_neuron_ratios.keys())
        ratios = list(dead_neuron_ratios.values())
        
        bars = ax.bar(range(len(layer_names)), ratios, alpha=0.7)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels([name.split('.')[-1] for name in layer_names], 
                          rotation=45, ha='right')
        ax.set_ylabel('Dead Neuron Ratio')
        ax.set_title('Dead Neurons Analysis Across Layers')
        ax.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 高亮问题层
        for i, ratio in enumerate(ratios):
            if ratio > 0.5:
                bars[i].set_color('red')
        
        plt.tight_layout()
        return fig, dead_neuron_ratios


class WeightAnalyzer:
    """权重分析器"""
    
    @staticmethod
    def analyze_weight_distribution(model):
        """
        分析模型权重分布
        """
        # 收集所有权重
        all_weights = []
        layer_weights = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                layer_weights[name] = weights
        
        all_weights = np.array(all_weights)
        
        # 创建可视化
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 总体分布直方图
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(all_weights, bins=100, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Weight Distribution')
        ax1.axvline(0, color='r', linestyle='--', linewidth=2)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {all_weights.mean():.6f}\nStd: {all_weights.std():.6f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Q-Q图（检验正态性）
        ax2 = fig.add_subplot(gs[1, 0])
        from scipy import stats
        stats.probplot(all_weights[::100], dist="norm", plot=ax2)  # 采样以加速
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 层级权重范围
        ax3 = fig.add_subplot(gs[1, 1:])
        layer_names = list(layer_weights.keys())[:20]  # 前20层
        layer_stds = [layer_weights[name].std() for name in layer_names]
        layer_means = [layer_weights[name].mean() for name in layer_names]
        
        x = range(len(layer_names))
        ax3.errorbar(x, layer_means, yerr=layer_stds, fmt='o-', capsize=5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([name.split('.')[-2] for name in layer_names], 
                           rotation=45, ha='right')
        ax3.set_ylabel('Weight Value')
        ax3.set_title('Weight Mean ± Std by Layer')
        ax3.axhline(0, color='r', linestyle='--')
        ax3.grid(True, alpha=0.3)
        
        # 4. 权重稀疏性分析
        ax4 = fig.add_subplot(gs[2, 0])
        sparsity_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        sparsity_ratios = []
        for level in sparsity_levels:
            ratio = (np.abs(all_weights) < level).sum() / len(all_weights)
            sparsity_ratios.append(ratio * 100)
        
        ax4.plot(range(len(sparsity_levels)), sparsity_ratios, 'o-', linewidth=2, markersize=8)
        ax4.set_xticks(range(len(sparsity_levels)))
        ax4.set_xticklabels([f'{l:.0e}' for l in sparsity_levels], rotation=45)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Sparsity (%)')
        ax4.set_title('Weight Sparsity Analysis')
        ax4.grid(True, alpha=0.3)
        
        # 5. 权重热图（选择一层）
        ax5 = fig.add_subplot(gs[2, 1:])
        # 选择第一个卷积层
        for name, weights in layer_weights.items():
            if 'conv' in name.lower():
                # 重塑为2D以显示
                w = weights.reshape(-1, min(100, len(weights)//10))[:100]
                im = ax5.imshow(w, cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
                ax5.set_title(f'Weight Heatmap: {name.split(".")[-2]}')
                ax5.set_xlabel('Weight Index')
                ax5.set_ylabel('Neuron Index')
                plt.colorbar(im, ax=ax5)
                break
        
        plt.suptitle('Weight Distribution Analysis', fontsize=16, fontweight='bold')
        
        return fig
    
    @staticmethod
    def analyze_gradient_statistics(model):
        """
        分析梯度统计信息
        """
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.cpu().numpy().flatten()
                gradient_stats[name] = {
                    'mean': np.mean(grad),
                    'std': np.std(grad),
                    'min': np.min(grad),
                    'max': np.max(grad),
                    'norm': np.linalg.norm(grad)
                }
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        layer_names = list(gradient_stats.keys())
        
        # 1. 梯度均值
        means = [gradient_stats[name]['mean'] for name in layer_names]
        axes[0, 0].bar(range(len(layer_names)), means, alpha=0.7)
        axes[0, 0].set_title('Gradient Mean by Layer')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].axhline(0, color='r', linestyle='--')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 梯度标准差
        stds = [gradient_stats[name]['std'] for name in layer_names]
        axes[0, 1].bar(range(len(layer_names)), stds, alpha=0.7, color='orange')
        axes[0, 1].set_title('Gradient Std by Layer')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 梯度范围
        mins = [gradient_stats[name]['min'] for name in layer_names]
        maxs = [gradient_stats[name]['max'] for name in layer_names]
        x = range(len(layer_names))
        axes[1, 0].fill_between(x, mins, maxs, alpha=0.3)
        axes[1, 0].plot(x, mins, 'b-', label='Min')
        axes[1, 0].plot(x, maxs, 'r-', label='Max')
        axes[1, 0].set_title('Gradient Range by Layer')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 梯度范数
        norms = [gradient_stats[name]['norm'] for name in layer_names]
        axes[1, 1].semilogy(range(len(layer_names)), norms, 'o-', linewidth=2)
        axes[1, 1].set_title('Gradient Norm by Layer')
        axes[1, 1].set_ylabel('Norm (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            if ax.get_xlabel() == '':
                ax.set_xlabel('Layer Index')
        
        plt.tight_layout()
        return fig, gradient_stats


class SensitivityAnalyzer:
    """敏感性分析器"""
    
    @staticmethod
    def compute_input_sensitivity(model, input_image, target_keypoint):
        """
        计算输入对特定关键点的敏感性（Saliency Map）
        """
        model.eval()
        input_image.requires_grad = True
        
        # 前向传播
        output = model(input_image)
        
        if isinstance(output, dict):
            heatmaps = output['heatmaps']
        else:
            heatmaps = output
        
        # 计算目标关键点的响应
        target_score = heatmaps[0, target_keypoint].sum()
        
        # 反向传播
        model.zero_grad()
        target_score.backward()
        
        # 获取梯度（敏感性）
        sensitivity = input_image.grad[0].abs().max(dim=0)[0]
        
        return sensitivity.cpu().numpy()
    
    @staticmethod
    def visualize_sensitivity_map(image, sensitivity_map):
        """
        可视化敏感性图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        if isinstance(image, torch.Tensor):
            image_np = image[0].permute(1, 2, 0).cpu().numpy()
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        else:
            image_np = image
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 敏感性图
        axes[1].imshow(sensitivity_map, cmap='hot')
        axes[1].set_title('Sensitivity Map')
        axes[1].axis('off')
        
        # 叠加显示
        sensitivity_colored = cv2.applyColorMap(
            np.uint8(255 * sensitivity_map / sensitivity_map.max()),
            cv2.COLORMAP_JET
        )
        sensitivity_colored = cv2.cvtColor(sensitivity_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(
            np.uint8(255 * image_np), 0.6,
            sensitivity_colored, 0.4, 0
        )
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def occlusion_sensitivity(model, image, target_keypoint, patch_size=16, stride=8):
        """
        遮挡敏感性分析
        通过遮挡图像不同区域来分析模型对各区域的依赖
        """
        model.eval()
        device = next(model.parameters()).device
        
        _, _, h, w = image.shape
        
        # 创建敏感性图
        sensitivity_map = np.zeros((h, w))
        
        # 获取原始预测
        with torch.no_grad():
            original_output = model(image.to(device))
            if isinstance(original_output, dict):
                original_score = original_output['heatmaps'][0, target_keypoint].sum().item()
            else:
                original_score = original_output[0, target_keypoint].sum().item()
        
        # 遮挡不同区域
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                # 创建遮挡图像
                occluded_image = image.clone()
                occluded_image[:, :, i:i+patch_size, j:j+patch_size] = 0
                
                # 预测
                with torch.no_grad():
                    output = model(occluded_image.to(device))
                    if isinstance(output, dict):
                        score = output['heatmaps'][0, target_keypoint].sum().item()
                    else:
                        score = output[0, target_keypoint].sum().item()
                
                # 计算敏感性
                sensitivity = original_score - score
                sensitivity_map[i:i+patch_size, j:j+patch_size] = sensitivity
        
        return sensitivity_map


class UncertaintyAnalyzer:
    """不确定性分析器"""
    
    @staticmethod
    def monte_carlo_dropout_uncertainty(model, image, num_samples=30):
        """
        使用Monte Carlo Dropout估计预测不确定性
        """
        # 开启dropout
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        
        model.eval()
        model.apply(enable_dropout)
        
        predictions = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = model(image.to(device))
                if isinstance(output, dict):
                    pred = output['heatmaps']
                else:
                    pred = output
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        num_keypoints = min(8, predictions.shape[2])
        
        for i in range(num_keypoints):
            row = i // 4
            col = i % 4
            
            # 均值预测
            im = axes[row, col].imshow(mean_pred[0, i], cmap='hot')
            axes[row, col].set_title(f'Joint {i}: Mean')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
            
            # 在同一子图上叠加不确定性（用轮廓线表示）
            axes[row, col].contour(std_pred[0, i], colors='blue', alpha=0.5, linewidths=1)
        
        plt.suptitle('Prediction Uncertainty (MC Dropout)', fontsize=16)
        plt.tight_layout()
        
        return fig, mean_pred, std_pred


# 使用示例
if __name__ == '__main__':
    print("高级神经网络分析工具")
    print("这些工具可以帮助深入理解模型的内部工作机制")
    print("\n包含的分析方法:")
    print("1. 激活值分析")
    print("2. 死神经元检测")
    print("3. 权重分布分析")
    print("4. 梯度统计")
    print("5. 输入敏感性分析")
    print("6. 遮挡敏感性分析")
    print("7. 不确定性估计")
