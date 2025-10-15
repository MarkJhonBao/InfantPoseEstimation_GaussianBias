"""
Analyze COCO format dataset for preterm infant pose estimation
Provides statistics and visualizations
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import os
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze COCO dataset')
    parser.add_argument('--ann_file', type=str, required=True,
                        help='COCO annotation JSON file')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='output directory for analysis results')
    parser.add_argument('--create_pdf', action='store_true',
                        help='create PDF report')
    return parser.parse_args()


class DatasetAnalyzer:
    """Analyze COCO format dataset"""
    
    def __init__(self, ann_file):
        self.coco = COCO(ann_file)
        self.ann_file = ann_file
        self.stats = {}
        
    def analyze_basic_stats(self):
        """Analyze basic dataset statistics"""
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        n_images = len(self.coco.imgs)
        n_annotations = len(self.coco.anns)
        n_categories = len(self.coco.cats)
        
        self.stats['n_images'] = n_images
        self.stats['n_annotations'] = n_annotations
        self.stats['n_categories'] = n_categories
        
        print(f"Total images: {n_images}")
        print(f"Total annotations: {n_annotations}")
        print(f"Total categories: {n_categories}")
        print(f"Annotations per image: {n_annotations/n_images:.2f}")
        
    def analyze_keypoint_visibility(self):
        """Analyze keypoint visibility statistics"""
        print("\n" + "="*80)
        print("KEYPOINT VISIBILITY ANALYSIS")
        print("="*80)
        
        # Get keypoint names
        cat = self.coco.loadCats(1)[0]
        keypoint_names = cat['keypoints']
        n_keypoints = len(keypoint_names)
        
        # Count visibility for each keypoint
        visibility_count = defaultdict(lambda: {'visible': 0, 'occluded': 0, 'not_labeled': 0})
        
        for ann_id in self.coco.anns:
            ann = self.coco.anns[ann_id]
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            
            for i, (x, y, v) in enumerate(keypoints):
                if i >= n_keypoints:
                    break
                
                if v == 0:
                    visibility_count[keypoint_names[i]]['not_labeled'] += 1
                elif v == 1:
                    visibility_count[keypoint_names[i]]['occluded'] += 1
                elif v == 2:
                    visibility_count[keypoint_names[i]]['visible'] += 1
        
        # Print statistics
        print(f"\n{'Keypoint':<20} {'Visible':<12} {'Occluded':<12} {'Not Labeled':<12} {'Visibility %':<12}")
        print("-" * 80)
        
        for name in keypoint_names:
            total = sum(visibility_count[name].values())
            visible = visibility_count[name]['visible']
            occluded = visibility_count[name]['occluded']
            not_labeled = visibility_count[name]['not_labeled']
            vis_pct = (visible / total * 100) if total > 0 else 0
            
            print(f"{name:<20} {visible:<12} {occluded:<12} {not_labeled:<12} {vis_pct:<12.2f}")
        
        self.stats['visibility'] = dict(visibility_count)
        return visibility_count, keypoint_names
    
    def analyze_bounding_boxes(self):
        """Analyze bounding box statistics"""
        print("\n" + "="*80)
        print("BOUNDING BOX ANALYSIS")
        print("="*80)
        
        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        
        for ann_id in self.coco.anns:
            ann = self.coco.anns[ann_id]
            bbox = ann['bbox']
            
            w, h = bbox[2], bbox[3]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            aspect_ratios.append(w / h if h > 0 else 0)
        
        print(f"Width  - Mean: {np.mean(widths):.2f}, Std: {np.std(widths):.2f}, "
              f"Min: {np.min(widths):.2f}, Max: {np.max(widths):.2f}")
        print(f"Height - Mean: {np.mean(heights):.2f}, Std: {np.std(heights):.2f}, "
              f"Min: {np.min(heights):.2f}, Max: {np.max(heights):.2f}")
        print(f"Area   - Mean: {np.mean(areas):.2f}, Std: {np.std(areas):.2f}, "
              f"Min: {np.min(areas):.2f}, Max: {np.max(areas):.2f}")
        print(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
        
        self.stats['bbox'] = {
            'widths': widths,
            'heights': heights,
            'areas': areas,
            'aspect_ratios': aspect_ratios
        }
        
        return widths, heights, areas, aspect_ratios
    
    def analyze_image_sizes(self):
        """Analyze image size distribution"""
        print("\n" + "="*80)
        print("IMAGE SIZE ANALYSIS")
        print("="*80)
        
        image_sizes = defaultdict(int)
        
        for img_id in self.coco.imgs:
            img = self.coco.imgs[img_id]
            size = f"{img['width']}x{img['height']}"
            image_sizes[size] += 1
        
        print(f"Unique image sizes: {len(image_sizes)}")
        print("\nTop 5 image sizes:")
        sorted_sizes = sorted(image_sizes.items(), key=lambda x: x[1], reverse=True)
        for size, count in sorted_sizes[:5]:
            print(f"  {size}: {count} images ({count/len(self.coco.imgs)*100:.2f}%)")
        
        self.stats['image_sizes'] = dict(image_sizes)
    
    def analyze_keypoint_positions(self):
        """Analyze keypoint position distributions"""
        print("\n" + "="*80)
        print("KEYPOINT POSITION ANALYSIS")
        print("="*80)
        
        cat = self.coco.loadCats(1)[0]
        keypoint_names = cat['keypoints']
        n_keypoints = len(keypoint_names)
        
        # Collect positions (normalized by bbox)
        keypoint_positions = defaultdict(lambda: {'x': [], 'y': []})
        
        for ann_id in self.coco.anns:
            ann = self.coco.anns[ann_id]
            bbox = ann['bbox']
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            
            for i, (x, y, v) in enumerate(keypoints):
                if i >= n_keypoints or v == 0:
                    continue
                
                # Normalize by bbox
                norm_x = (x - bbox[0]) / bbox[2] if bbox[2] > 0 else 0
                norm_y = (y - bbox[1]) / bbox[3] if bbox[3] > 0 else 0
                
                keypoint_positions[keypoint_names[i]]['x'].append(norm_x)
                keypoint_positions[keypoint_names[i]]['y'].append(norm_y)
        
        print("\nNormalized keypoint positions (relative to bbox):")
        print(f"{'Keypoint':<20} {'Mean X':<10} {'Std X':<10} {'Mean Y':<10} {'Std Y':<10}")
        print("-" * 60)
        
        for name in keypoint_names:
            if len(keypoint_positions[name]['x']) > 0:
                mean_x = np.mean(keypoint_positions[name]['x'])
                std_x = np.std(keypoint_positions[name]['x'])
                mean_y = np.mean(keypoint_positions[name]['y'])
                std_y = np.std(keypoint_positions[name]['y'])
                
                print(f"{name:<20} {mean_x:<10.3f} {std_x:<10.3f} {mean_y:<10.3f} {std_y:<10.3f}")
        
        self.stats['keypoint_positions'] = dict(keypoint_positions)
        return keypoint_positions, keypoint_names
    
    def plot_visualizations(self, output_dir):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Bounding box distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        bbox_stats = self.stats['bbox']
        
        axes[0, 0].hist(bbox_stats['widths'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Bounding Box Width Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(bbox_stats['heights'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bounding Box Height Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].hist(bbox_stats['areas'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Area (pixels²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Bounding Box Area Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(bbox_stats['aspect_ratios'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Aspect Ratio (W/H)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Aspect Ratio Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bbox_distribution.png'), dpi=150)
        plt.close()
        
        # 2. Keypoint visibility
        visibility_data = self.stats['visibility']
        cat = self.coco.loadCats(1)[0]
        keypoint_names = cat['keypoints']
        
        visible_counts = [visibility_data[name]['visible'] for name in keypoint_names]
        occluded_counts = [visibility_data[name]['occluded'] for name in keypoint_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(keypoint_names))
        width = 0.35
        
        ax.bar(x - width/2, visible_counts, width, label='Visible', color='lightgreen')
        ax.bar(x + width/2, occluded_counts, width, label='Occluded', color='lightcoral')
        
        ax.set_xlabel('Keypoint')
        ax.set_ylabel('Count')
        ax.set_title('Keypoint Visibility Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(keypoint_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keypoint_visibility.png'), dpi=150)
        plt.close()
        
        # 3. Keypoint position heatmap
        positions = self.stats['keypoint_positions']
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(keypoint_names):
            if i >= len(axes):
                break
            
            if len(positions[name]['x']) > 0:
                x_pos = positions[name]['x']
                y_pos = positions[name]['y']
                
                # Create 2D histogram
                h, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=20, range=[[0, 1], [0, 1]])
                
                im = axes[i].imshow(h.T, origin='lower', extent=[0, 1, 0, 1], 
                                   cmap='hot', interpolation='bilinear')
                axes[i].set_title(name, fontsize=10)
                axes[i].set_xlim([0, 1])
                axes[i].set_ylim([0, 1])
                axes[i].invert_yaxis()
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i].set_xlim([0, 1])
                axes[i].set_ylim([0, 1])
        
        # Hide extra subplots
        for i in range(len(keypoint_names), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Keypoint Position Heatmaps (Normalized by BBox)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keypoint_positions.png'), dpi=150)
        plt.close()
        
        print(f"\nVisualizations saved to: {output_dir}")
    
    def generate_report(self, output_dir):
        """Generate comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'dataset_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PRETERM INFANT POSE DATASET ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {self.ann_file}\n\n")
            
            # Basic stats
            f.write("BASIC STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total images: {self.stats['n_images']}\n")
            f.write(f"Total annotations: {self.stats['n_annotations']}\n")
            f.write(f"Annotations per image: {self.stats['n_annotations']/self.stats['n_images']:.2f}\n\n")
            
            # Bounding box stats
            f.write("BOUNDING BOX STATISTICS\n")
            f.write("-"*80 + "\n")
            bbox = self.stats['bbox']
            f.write(f"Width  - Mean: {np.mean(bbox['widths']):.2f}, Std: {np.std(bbox['widths']):.2f}\n")
            f.write(f"Height - Mean: {np.mean(bbox['heights']):.2f}, Std: {np.std(bbox['heights']):.2f}\n")
            f.write(f"Area   - Mean: {np.mean(bbox['areas']):.2f}, Std: {np.std(bbox['areas']):.2f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            # Check for data quality issues
            if np.min(bbox['areas']) < 1000:
                f.write("⚠ Warning: Very small bounding boxes detected. Consider reviewing annotations.\n")
            
            if np.std(bbox['aspect_ratios']) > 0.5:
                f.write("⚠ Warning: High variance in aspect ratios. Consider aspect ratio augmentation.\n")
            
            # Check keypoint visibility
            cat = self.coco.loadCats(1)[0]
            for name in cat['keypoints']:
                total = sum(self.stats['visibility'][name].values())
                visible = self.stats['visibility'][name]['visible']
                vis_rate = visible / total if total > 0 else 0
                
                if vis_rate < 0.5:
                    f.write(f"⚠ Warning: Low visibility rate for '{name}': {vis_rate*100:.1f}%\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
        
        print(f"\nReport saved to: {report_path}")


def main():
    args = parse_args()
    
    print(f"Analyzing dataset: {args.ann_file}")
    
    # Create analyzer
    analyzer = DatasetAnalyzer(args.ann_file)
    
    # Run analyses
    analyzer.analyze_basic_stats()
    analyzer.analyze_keypoint_visibility()
    analyzer.analyze_bounding_boxes()
    analyzer.analyze_image_sizes()
    analyzer.analyze_keypoint_positions()
    
    # Generate visualizations
    analyzer.plot_visualizations(args.output_dir)
    
    # Generate report
    analyzer.generate_report(args.output_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
