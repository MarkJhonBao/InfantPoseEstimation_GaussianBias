"""
Convert custom annotations to COCO format for preterm infant pose estimation
Supports various input formats
"""
import json
import os
import argparse
from datetime import datetime
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert annotations to COCO format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='directory containing raw annotations')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='directory containing images')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output COCO JSON file path')
    parser.add_argument('--format', type=str, default='custom',
                        choices=['custom', 'json', 'txt'],
                        help='input annotation format')
    parser.add_argument('--num_keypoints', type=int, default=13,
                        help='number of keypoints')
    return parser.parse_args()


class COCOConverter:
    """Convert annotations to COCO format"""
    
    def __init__(self, num_keypoints=13):
        self.num_keypoints = num_keypoints
        
        # COCO format structure
        self.coco_format = {
            'info': {
                'description': 'Preterm Infant Pose Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().strftime('%Y-%m-%d')
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': self._get_categories()
        }
        
        self.image_id = 1
        self.annotation_id = 1
    
    def _get_categories(self):
        """Define keypoint categories"""
        return [{
            'id': 1,
            'name': 'preterm_infant',
            'supercategory': 'person',
            'keypoints': [
                'nose', 'left_eye', 'right_eye',
                'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist',
                'left_hip', 'right_hip'
            ],
            'skeleton': [
                [0, 1], [0, 2],           # nose to eyes
                [1, 3], [2, 4],           # eyes to ears
                [5, 6],                   # shoulders
                [5, 7], [7, 9],           # left arm
                [6, 8], [8, 10],          # right arm
                [5, 11], [6, 12],         # shoulder to hip
                [11, 12]                  # hips
            ]
        }]
    
    def add_image(self, image_path, file_name):
        """Add image to COCO format"""
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        height, width = img.shape[:2]
        
        image_info = {
            'id': self.image_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'date_captured': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.coco_format['images'].append(image_info)
        current_id = self.image_id
        self.image_id += 1
        
        return current_id
    
    def add_annotation(self, image_id, keypoints, bbox=None, segmentation=None):
        """
        Add annotation to COCO format
        
        Args:
            image_id: image ID
            keypoints: list of [x, y, v] for each keypoint
            bbox: [x, y, width, height]
            segmentation: segmentation mask (optional)
        """
        # Calculate bbox from keypoints if not provided
        if bbox is None:
            visible_kpts = [(x, y) for x, y, v in keypoints if v > 0]
            if len(visible_kpts) == 0:
                return
            
            xs, ys = zip(*visible_kpts)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Add padding
            padding = 20
            bbox = [
                max(0, x_min - padding),
                max(0, y_min - padding),
                x_max - x_min + 2 * padding,
                y_max - y_min + 2 * padding
            ]
        
        # Flatten keypoints
        keypoints_flat = []
        num_visible = 0
        for x, y, v in keypoints:
            keypoints_flat.extend([x, y, v])
            if v > 0:
                num_visible += 1
        
        annotation = {
            'id': self.annotation_id,
            'image_id': image_id,
            'category_id': 1,
            'keypoints': keypoints_flat,
            'num_keypoints': num_visible,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0
        }
        
        if segmentation is not None:
            annotation['segmentation'] = segmentation
        
        self.coco_format['annotations'].append(annotation)
        self.annotation_id += 1
    
    def parse_custom_format(self, annotation_file):
        """
        Parse custom annotation format
        
        Expected format: JSON with structure:
        {
            "image": "filename.jpg",
            "keypoints": [[x1, y1, v1], [x2, y2, v2], ...]
        }
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        return {
            'file_name': data['image'],
            'keypoints': data['keypoints'],
            'bbox': data.get('bbox', None)
        }
    
    def parse_txt_format(self, annotation_file):
        """
        Parse text annotation format
        
        Expected format:
        filename.jpg
        x1 y1 v1
        x2 y2 v2
        ...
        """
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        file_name = lines[0].strip()
        keypoints = []
        
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y, v = float(parts[0]), float(parts[1]), int(parts[2])
                keypoints.append([x, y, v])
        
        return {
            'file_name': file_name,
            'keypoints': keypoints,
            'bbox': None
        }
    
    def save(self, output_path):
        """Save COCO format JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_format, f, indent=2)
        
        print(f"\nCOCO format annotation saved to: {output_path}")
        print(f"Total images: {len(self.coco_format['images'])}")
        print(f"Total annotations: {len(self.coco_format['annotations'])}")


def convert_dataset(args):
    """Convert entire dataset to COCO format"""
    converter = COCOConverter(num_keypoints=args.num_keypoints)
    
    # Get annotation files
    if args.format == 'json':
        ann_files = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    elif args.format == 'txt':
        ann_files = [f for f in os.listdir(args.input_dir) if f.endswith('.txt')]
    else:
        ann_files = [f for f in os.listdir(args.input_dir) 
                     if f.endswith(('.json', '.txt'))]
    
    print(f"Found {len(ann_files)} annotation files")
    
    for ann_file in tqdm(ann_files, desc="Converting annotations"):
        ann_path = os.path.join(args.input_dir, ann_file)
        
        try:
            # Parse annotation
            if args.format == 'txt':
                ann_data = converter.parse_txt_format(ann_path)
            else:
                ann_data = converter.parse_custom_format(ann_path)
            
            # Get image path
            image_path = os.path.join(args.image_dir, ann_data['file_name'])
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Add image
            image_id = converter.add_image(image_path, ann_data['file_name'])
            
            if image_id is None:
                continue
            
            # Add annotation
            converter.add_annotation(
                image_id,
                ann_data['keypoints'],
                ann_data.get('bbox')
            )
            
        except Exception as e:
            print(f"Error processing {ann_file}: {str(e)}")
            continue
    
    # Save COCO format
    converter.save(args.output_file)


def create_sample_annotation():
    """Create sample annotation file for reference"""
    sample = {
        "image": "infant_001.jpg",
        "keypoints": [
            [320, 100, 2],  # nose
            [310, 95, 2],   # left_eye
            [330, 95, 2],   # right_eye
            [300, 100, 2],  # left_ear
            [340, 100, 2],  # right_ear
            [280, 150, 2],  # left_shoulder
            [360, 150, 2],  # right_shoulder
            [260, 200, 2],  # left_elbow
            [380, 200, 2],  # right_elbow
            [250, 250, 2],  # left_wrist
            [390, 250, 2],  # right_wrist
            [300, 300, 2],  # left_hip
            [340, 300, 2]   # right_hip
        ],
        "bbox": [200, 50, 250, 300]
    }
    
    with open('sample_annotation.json', 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("Sample annotation created: sample_annotation.json")
    print("\nFormat:")
    print("- keypoints: [[x, y, visibility], ...] where visibility: 0=not labeled, 1=labeled but occluded, 2=labeled and visible")
    print("- bbox: [x, y, width, height] (optional, will be calculated from keypoints if not provided)")


def validate_coco_format(coco_file):
    """Validate COCO format JSON"""
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    print("\n=== COCO Format Validation ===")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    # Check required fields
    required_keys = ['info', 'images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            print(f"ERROR: Missing required key: {key}")
            return False
    
    # Check image fields
    if len(data['images']) > 0:
        img = data['images'][0]
        required_img_keys = ['id', 'file_name', 'height', 'width']
        for key in required_img_keys:
            if key not in img:
                print(f"ERROR: Image missing required key: {key}")
                return False
    
    # Check annotation fields
    if len(data['annotations']) > 0:
        ann = data['annotations'][0]
        required_ann_keys = ['id', 'image_id', 'category_id', 'keypoints', 'num_keypoints', 'bbox']
        for key in required_ann_keys:
            if key not in ann:
                print(f"ERROR: Annotation missing required key: {key}")
                return False
        
        # Check keypoints format
        num_kpts = len(ann['keypoints']) // 3
        print(f"Number of keypoints: {num_kpts}")
        print(f"Number of visible keypoints: {ann['num_keypoints']}")
    
    print("✓ COCO format validation passed")
    return True


def split_dataset(coco_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split COCO dataset into train/val/test"""
    import random
    
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    # Shuffle images
    images = data['images']
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    # Split annotations
    def create_split(split_images):
        split_img_ids = {img['id'] for img in split_images}
        split_anns = [ann for ann in data['annotations'] 
                      if ann['image_id'] in split_img_ids]
        
        return {
            'info': data['info'],
            'licenses': data['licenses'],
            'images': split_images,
            'annotations': split_anns,
            'categories': data['categories']
        }
    
    # Save splits
    base_dir = os.path.dirname(coco_file)
    base_name = os.path.splitext(os.path.basename(coco_file))[0]
    
    splits = {
        'train': create_split(train_images),
        'val': create_split(val_images),
        'test': create_split(test_images)
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(base_dir, f'{base_name}_{split_name}.json')
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"{split_name}: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations")
        print(f"  Saved to: {output_path}")


def main():
    args = parse_args()
    
    print("Converting annotations to COCO format...")
    print(f"Input directory: {args.input_dir}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Format: {args.format}")
    
    convert_dataset(args)
    
    # Validate output
    if os.path.exists(args.output_file):
        validate_coco_format(args.output_file)


if __name__ == '__main__':
    # Uncomment to create sample annotation
    # create_sample_annotation()
    
    main()
