"""
COCO数据集关键点扩展工具
支持添加任意数量的自定义关键点，例如68个面部关键点

功能：
1. 扩展COCO格式支持更多关键点
2. 预定义模板（面部68点、手部21点等）
3. 数据集合并工具
4. 兼容性验证
"""

import json
import numpy as np
import argparse
from collections import OrderedDict
import os


class COCOKeypointExtender:
    """COCO关键点扩展器"""
    
    # 预定义的关键点模板
    TEMPLATES = {
        'face_68': {
            'num_keypoints': 68,
            'names': [
                # 下巴轮廓 (0-16)
                'jaw_0', 'jaw_1', 'jaw_2', 'jaw_3', 'jaw_4', 'jaw_5', 'jaw_6', 'jaw_7',
                'jaw_8', 'jaw_9', 'jaw_10', 'jaw_11', 'jaw_12', 'jaw_13', 'jaw_14', 'jaw_15', 'jaw_16',
                # 左眉毛 (17-21)
                'left_eyebrow_0', 'left_eyebrow_1', 'left_eyebrow_2', 'left_eyebrow_3', 'left_eyebrow_4',
                # 右眉毛 (22-26)
                'right_eyebrow_0', 'right_eyebrow_1', 'right_eyebrow_2', 'right_eyebrow_3', 'right_eyebrow_4',
                # 鼻梁 (27-30)
                'nose_bridge_0', 'nose_bridge_1', 'nose_bridge_2', 'nose_bridge_3',
                # 鼻尖 (31-35)
                'nose_tip_0', 'nose_tip_1', 'nose_tip_2', 'nose_tip_3', 'nose_tip_4',
                # 左眼 (36-41)
                'left_eye_0', 'left_eye_1', 'left_eye_2', 'left_eye_3', 'left_eye_4', 'left_eye_5',
                # 右眼 (42-47)
                'right_eye_0', 'right_eye_1', 'right_eye_2', 'right_eye_3', 'right_eye_4', 'right_eye_5',
                # 外嘴唇 (48-59)
                'outer_lip_0', 'outer_lip_1', 'outer_lip_2', 'outer_lip_3', 'outer_lip_4', 'outer_lip_5',
                'outer_lip_6', 'outer_lip_7', 'outer_lip_8', 'outer_lip_9', 'outer_lip_10', 'outer_lip_11',
                # 内嘴唇 (60-67)
                'inner_lip_0', 'inner_lip_1', 'inner_lip_2', 'inner_lip_3',
                'inner_lip_4', 'inner_lip_5', 'inner_lip_6', 'inner_lip_7'
            ],
            'skeleton': [
                # 下巴轮廓
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
                # 左眉毛
                [17, 18], [18, 19], [19, 20], [20, 21],
                # 右眉毛
                [22, 23], [23, 24], [24, 25], [25, 26],
                # 鼻梁
                [27, 28], [28, 29], [29, 30],
                # 鼻尖
                [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [35, 31],
                # 左眼
                [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
                # 右眼
                [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
                # 外嘴唇
                [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
                [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
                # 内嘴唇
                [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
            ]
        },
        'hand_21': {
            'num_keypoints': 21,
            'names': [
                'wrist',
                # 拇指
                'thumb_1', 'thumb_2', 'thumb_3', 'thumb_4',
                # 食指
                'index_1', 'index_2', 'index_3', 'index_4',
                # 中指
                'middle_1', 'middle_2', 'middle_3', 'middle_4',
                # 无名指
                'ring_1', 'ring_2', 'ring_3', 'ring_4',
                # 小指
                'pinky_1', 'pinky_2', 'pinky_3', 'pinky_4'
            ],
            'skeleton': [
                # 手腕到各手指根部
                [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
                # 拇指
                [1, 2], [2, 3], [3, 4],
                # 食指
                [5, 6], [6, 7], [7, 8],
                # 中指
                [9, 10], [10, 11], [11, 12],
                # 无名指
                [13, 14], [14, 15], [15, 16],
                # 小指
                [17, 18], [18, 19], [19, 20]
            ]
        },
        'body_coco_17': {
            'num_keypoints': 17,
            'names': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'skeleton': [
                [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
                [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                [13, 15], [12, 14], [14, 16]
            ]
        },
        'preemie_infant_13': {
            'num_keypoints': 13,
            'names': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip'
            ],
            'skeleton': [
                [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
                [6, 8], [8, 10], [5, 11], [6, 12], [11, 12]
            ]
        }
    }
    
    def __init__(self, base_coco_file=None):
        """
        初始化扩展器
        
        Args:
            base_coco_file: 基础COCO文件路径（可选）
        """
        if base_coco_file:
            with open(base_coco_file, 'r') as f:
                self.coco_data = json.load(f)
        else:
            self.coco_data = self._create_empty_coco()
    
    def _create_empty_coco(self):
        """创建空的COCO格式数据"""
        return {
            'info': {
                'description': 'Extended COCO Keypoint Dataset',
                'version': '1.0',
                'year': 2025
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
    
    def add_keypoint_category(self, category_id, category_name, template_name=None, 
                             custom_keypoints=None, custom_skeleton=None):
        """
        添加新的关键点类别
        
        Args:
            category_id: 类别ID
            category_name: 类别名称
            template_name: 使用预定义模板 ('face_68', 'hand_21', etc.)
            custom_keypoints: 自定义关键点名称列表
            custom_skeleton: 自定义骨架连接
        """
        if template_name and template_name in self.TEMPLATES:
            template = self.TEMPLATES[template_name]
            keypoint_names = template['names']
            skeleton = template['skeleton']
        elif custom_keypoints:
            keypoint_names = custom_keypoints
            skeleton = custom_skeleton if custom_skeleton else []
        else:
            raise ValueError("必须指定template_name或custom_keypoints")
        
        category = {
            'id': category_id,
            'name': category_name,
            'supercategory': 'person',
            'keypoints': keypoint_names,
            'skeleton': skeleton
        }
        
        self.coco_data['categories'].append(category)
        
        print(f"✓ 添加类别: {category_name}")
        print(f"  关键点数量: {len(keypoint_names)}")
        print(f"  骨架连接数: {len(skeleton)}")
        
        return category
    
    def merge_keypoint_categories(self, categories_to_merge):
        """
        合并多个关键点类别（例如：身体+面部+手部）
        
        Args:
            categories_to_merge: 要合并的类别列表
        
        Returns:
            merged_category: 合并后的类别
        """
        merged_keypoints = []
        merged_skeleton = []
        offset = 0
        
        for cat_name in categories_to_merge:
            if cat_name not in self.TEMPLATES:
                raise ValueError(f"未找到模板: {cat_name}")
            
            template = self.TEMPLATES[cat_name]
            
            # 添加关键点（带前缀）
            prefix = cat_name.split('_')[0]
            for kp_name in template['names']:
                merged_keypoints.append(f"{prefix}_{kp_name}")
            
            # 添加骨架（调整索引）
            for connection in template['skeleton']:
                merged_skeleton.append([connection[0] + offset, connection[1] + offset])
            
            offset += template['num_keypoints']
        
        merged_category = {
            'id': 1,
            'name': 'full_body_face_hands',
            'supercategory': 'person',
            'keypoints': merged_keypoints,
            'skeleton': merged_skeleton
        }
        
        print(f"✓ 合并类别完成")
        print(f"  总关键点数: {len(merged_keypoints)}")
        print(f"  组成: {' + '.join(categories_to_merge)}")
        
        return merged_category
    
    def add_annotation(self, image_id, category_id, keypoints, bbox=None, 
                      segmentation=None, area=None):
        """
        添加关键点标注
        
        Args:
            image_id: 图像ID
            category_id: 类别ID
            keypoints: 关键点列表 [[x1,y1,v1], [x2,y2,v2], ...]
            bbox: 边界框 [x, y, width, height]
            segmentation: 分割mask（可选）
            area: 区域面积（可选）
        """
        # 展平关键点列表
        keypoints_flat = []
        num_visible = 0
        for x, y, v in keypoints:
            keypoints_flat.extend([x, y, v])
            if v > 0:
                num_visible += 1
        
        # 如果没有提供bbox，从关键点计算
        if bbox is None:
            visible_kpts = [(x, y) for x, y, v in keypoints if v > 0]
            if visible_kpts:
                xs, ys = zip(*visible_kpts)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                padding = 20
                bbox = [x_min - padding, y_min - padding, 
                       x_max - x_min + 2*padding, y_max - y_min + 2*padding]
            else:
                bbox = [0, 0, 0, 0]
        
        if area is None:
            area = bbox[2] * bbox[3]
        
        annotation = {
            'id': len(self.coco_data['annotations']) + 1,
            'image_id': image_id,
            'category_id': category_id,
            'keypoints': keypoints_flat,
            'num_keypoints': num_visible,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        }
        
        if segmentation:
            annotation['segmentation'] = segmentation
        
        self.coco_data['annotations'].append(annotation)
        
        return annotation
    
    def save(self, output_path):
        """保存扩展后的COCO数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"\n✓ 保存成功: {output_path}")
        print(f"  图像数: {len(self.coco_data['images'])}")
        print(f"  标注数: {len(self.coco_data['annotations'])}")
        print(f"  类别数: {len(self.coco_data['categories'])}")
    
    @staticmethod
    def validate_keypoint_format(coco_file):
        """验证COCO关键点格式是否正确"""
        with open(coco_file, 'r') as f:
            data = json.load(f)
        
        print("="*80)
        print("COCO关键点格式验证")
        print("="*80)
        
        # 检查必要字段
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                print(f"✗ 缺少必要字段: {key}")
                return False
            else:
                print(f"✓ 字段存在: {key}")
        
        # 检查类别
        print(f"\n类别信息:")
        for cat in data['categories']:
            print(f"  ID: {cat['id']}, 名称: {cat['name']}")
            print(f"  关键点数量: {len(cat['keypoints'])}")
            print(f"  骨架连接数: {len(cat.get('skeleton', []))}")
            
            # 验证关键点数量
            if 'keypoints' not in cat:
                print(f"  ✗ 缺少keypoints字段")
                return False
        
        # 检查标注
        print(f"\n标注验证:")
        for ann in data['annotations'][:3]:  # 只检查前3个
            keypoints = ann['keypoints']
            num_values = len(keypoints)
            
            if num_values % 3 != 0:
                print(f"  ✗ 标注ID {ann['id']}: 关键点数量不是3的倍数")
                return False
            
            num_keypoints = num_values // 3
            declared_num = ann.get('num_keypoints', 0)
            
            # 计算实际可见关键点
            actual_visible = sum(1 for i in range(2, num_values, 3) if keypoints[i] > 0)
            
            print(f"  标注ID {ann['id']}:")
            print(f"    关键点总数: {num_keypoints}")
            print(f"    声明可见数: {declared_num}")
            print(f"    实际可见数: {actual_visible}")
            
            if declared_num != actual_visible:
                print(f"    ⚠ 警告: 声明数与实际数不匹配")
        
        print("\n✓ 格式验证通过")
        return True
    
    @staticmethod
    def visualize_keypoint_template(template_name, output_path=None):
        """可视化关键点模板"""
        import matplotlib.pyplot as plt
        
        if template_name not in COCOKeypointExtender.TEMPLATES:
            print(f"✗ 未找到模板: {template_name}")
            return
        
        template = COCOKeypointExtender.TEMPLATES[template_name]
        keypoints = template['names']
        skeleton = template['skeleton']
        
        # 创建模拟关键点位置
        num_kpts = len(keypoints)
        
        if template_name == 'face_68':
            # 面部68点的近似布局
            coords = np.zeros((num_kpts, 2))
            # 这里可以添加更精确的布局
            for i in range(num_kpts):
                angle = 2 * np.pi * i / num_kpts
                coords[i] = [100 + 50 * np.cos(angle), 100 + 50 * np.sin(angle)]
        else:
            # 其他类型使用简单布局
            coords = np.random.rand(num_kpts, 2) * 200
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制骨架
        for connection in skeleton:
            pt1 = coords[connection[0]]
            pt2 = coords[connection[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2, alpha=0.5)
        
        # 绘制关键点
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5)
        
        # 添加标签
        for i, (x, y) in enumerate(coords):
            ax.annotate(f'{i}', (x, y), fontsize=8, ha='center', va='center',
                       color='white', weight='bold')
        
        ax.set_title(f'Keypoint Template: {template_name}\n{num_kpts} keypoints', 
                    fontsize=16)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 模板可视化已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()


def create_face_68_example():
    """创建面部68关键点的示例数据集"""
    print("\n" + "="*80)
    print("创建面部68关键点示例数据集")
    print("="*80)
    
    extender = COCOKeypointExtender()
    
    # 添加面部68点类别
    extender.add_keypoint_category(
        category_id=1,
        category_name='face_68_landmarks',
        template_name='face_68'
    )
    
    # 添加示例图像
    extender.coco_data['images'].append({
        'id': 1,
        'file_name': 'face_sample_001.jpg',
        'height': 480,
        'width': 640
    })
    
    # 生成模拟的68个关键点（实际应该是真实标注）
    fake_keypoints = []
    center_x, center_y = 320, 240
    for i in range(68):
        angle = 2 * np.pi * i / 68
        x = center_x + 80 * np.cos(angle)
        y = center_y + 80 * np.sin(angle)
        v = 2  # 可见
        fake_keypoints.append([x, y, v])
    
    # 添加标注
    extender.add_annotation(
        image_id=1,
        category_id=1,
        keypoints=fake_keypoints
    )
    
    # 保存
    extender.save('face_68_example.json')
    
    # 验证
    COCOKeypointExtender.validate_keypoint_format('face_68_example.json')
    
    return extender


def create_merged_body_face_hands():
    """创建身体+面部+手部的完整关键点数据集"""
    print("\n" + "="*80)
    print("创建完整人体关键点数据集（身体17 + 面部68 + 左手21 + 右手21 = 127点）")
    print("="*80)
    
    extender = COCOKeypointExtender()
    
    # 方法1: 直接合并模板
    merged_cat = extender.merge_keypoint_categories([
        'body_coco_17',
        'face_68',
        'hand_21',  # 左手
        'hand_21'   # 右手（会自动加前缀区分）
    ])
    
    extender.coco_data['categories'].append(merged_cat)
    
    # 添加示例数据
    extender.coco_data['images'].append({
        'id': 1,
        'file_name': 'full_body_001.jpg',
        'height': 1080,
        'width': 1920
    })
    
    # 生成模拟关键点（127个）
    num_total_kpts = 17 + 68 + 21 + 21
    fake_keypoints = [[960 + i*5, 540 + i*3, 2] for i in range(num_total_kpts)]
    
    extender.add_annotation(
        image_id=1,
        category_id=1,
        keypoints=fake_keypoints
    )
    
    extender.save('full_body_face_hands.json')
    
    return extender


def convert_existing_to_extended(input_coco_file, output_file, new_template):
    """将现有COCO数据集转换为扩展格式"""
    print(f"\n转换数据集: {input_coco_file}")
    print(f"目标格式: {new_template}")
    
    with open(input_coco_file, 'r') as f:
        data = json.load(f)
    
    # 检查是否需要转换
    old_num_kpts = len(data['categories'][0]['keypoints'])
    new_num_kpts = COCOKeypointExtender.TEMPLATES[new_template]['num_keypoints']
    
    if old_num_kpts >= new_num_kpts:
        print("⚠ 警告: 目标格式关键点数不多于源格式")
        return
    
    print(f"扩展: {old_num_kpts} → {new_num_kpts} 关键点")
    
    # 创建扩展器
    extender = COCOKeypointExtender()
    extender.coco_data = data
    
    # 替换类别定义
    extender.coco_data['categories'] = []
    extender.add_keypoint_category(1, 'extended_keypoints', template_name=new_template)
    
    # 扩展每个标注（填充额外的关键点为不可见）
    for ann in extender.coco_data['annotations']:
        old_kpts = ann['keypoints']
        # 添加额外的不可见关键点
        for _ in range(new_num_kpts - old_num_kpts):
            old_kpts.extend([0, 0, 0])
        ann['keypoints'] = old_kpts
    
    extender.save(output_file)
    print(f"✓ 转换完成")


# CLI接口
def main():
    parser = argparse.ArgumentParser(description='COCO关键点扩展工具')
    parser.add_argument('--action', type=str, required=True,
                       choices=['create_face68', 'create_merged', 'convert', 'validate', 'visualize'],
                       help='操作类型')
    parser.add_argument('--input', type=str, help='输入COCO文件')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--template', type=str, help='模板名称')
    
    args = parser.parse_args()
    
    if args.action == 'create_face68':
        create_face_68_example()
    
    elif args.action == 'create_merged':
        create_merged_body_face_hands()
    
    elif args.action == 'convert':
        if not args.input or not args.output or not args.template:
            print("✗ convert操作需要--input, --output, --template参数")
            return
        convert_existing_to_extended(args.input, args.output, args.template)
    
    elif args.action == 'validate':
        if not args.input:
            print("✗ validate操作需要--input参数")
            return
        COCOKeypointExtender.validate_keypoint_format(args.input)
    
    elif args.action == 'visualize':
        if not args.template:
            print("✗ visualize操作需要--template参数")
            return
        COCOKeypointExtender.visualize_keypoint_template(
            args.template,
            args.output
        )


if __name__ == '__main__':
    # 演示用法
    print("COCO关键点扩展工具")
    print("="*80)
    print("\n可用模板:")
    for name, template in COCOKeypointExtender.TEMPLATES.items():
        print(f"  {name}: {template['num_keypoints']} 关键点")
    
    print("\n使用示例:")
    print("1. 创建面部68点数据集:")
    print("   python extend_coco_keypoints.py --action create_face68")
    print("\n2. 创建完整身体数据集:")
    print("   python extend_coco_keypoints.py --action create_merged")
    print("\n3. 转换现有数据集:")
    print("   python extend_coco_keypoints.py --action convert --input old.json --output new.json --template face_68")
    print("\n4. 验证数据集:")
    print("   python extend_coco_keypoints.py --action validate --input dataset.json")
    print("\n5. 可视化模板:")
    print("   python extend_coco_keypoints.py --action visualize --template face_68 --output template.png")
    
    # 如果直接运行，创建示例
    print("\n" + "="*80)
    print("运行示例...")
    print("="*80)
    
    # 示例1: 面部68点
    create_face_68_example()
    
    # 示例2: 完整身体
    # create_merged_body_face_hands()
