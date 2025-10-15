# Preterm Infant Limb Movement Recognition (PI-LMR)

## 📋 Overview

A top-down pose estimation framework for continuous monitoring of preterm infants' limb movements in NICUs. This system combines fused heatmap and regression-based representations with morphology-aware shape constraints to achieve robust and accurate pose detection.

**Key Features:**
- ✨ **Fused Head Architecture**: Combines heatmap-based and regression-based pose representations
- 🎯 **Morphology-Aware Loss**: Enforces distribution consistency for stable keypoint detection
- 🔧 **Coordinate Refinement**: Joint optimization to reduce localization errors
- 📊 **AP: 95.4%** on Infant-Skeleton-V2 dataset

## 🏗️ Project Structure

```
preterm-infant-pose/
├── train.py                    # Main training script
├── inference.py                # Inference and visualization
├── config.py                   # Configuration management
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── configs/                    # Configuration files
│   ├── default.yaml
│   └── hrnet_w32.yaml
│
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── pose_hrnet.py          # HRNet backbone with fused head
│   ├── losses.py              # Loss functions
│   └── layers.py              # Custom layers
│
├── data/                       # Data loading and processing
│   ├── __init__.py
│   ├── coco_dataset.py        # COCO format dataset loader
│   ├── transforms.py          # Data augmentation
│   └── generate_heatmap.py    # Heatmap generation utilities
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics (AP, PCK)
│   ├── postprocess.py         # Fused decoding and refinement
│   ├── visualization.py       # Visualization tools
│   └── logger.py              # Logging utilities
│
└── tools/                      # Helper scripts
    ├── convert_to_coco.py     # Convert annotations to COCO format
    └── analyze_dataset.py      # Dataset statistics

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/preterm-infant-pose.git
cd preterm-infant-pose

# Create virtual environment
conda create -n pi_pose python=3.8
conda activate pi_pose

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your dataset in COCO format:

```
data/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    └── val/
```

**COCO Annotation Format for Preterm Infants:**

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "infant_001.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "keypoints": [x1,y1,v1, x2,y2,v2, ...],
      "num_keypoints": 13
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "preterm_infant",
      "keypoints": [
        "nose", "left_eye", "right_eye", 
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip"
      ],
      "skeleton": [
        [0,1], [0,2], [1,3], [2,4],
        [5,6], [5,7], [7,9], [6,8], [8,10],
        [5,11], [6,12], [11,12]
      ]
    }
  ]
}
```

### 3. Training

```bash
# Train with default configuration
python train.py --data_dir ./data --output_dir ./outputs

# Train with custom config
python train.py \
    --config configs/hrnet_w32.yaml \
    --data_dir ./data \
    --output_dir ./outputs \
    --gpus 0,1

# Resume training
python train.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --resume ./outputs/checkpoint_epoch_50.pth
```

### 4. Inference

```bash
# Run inference on validation set
python inference.py \
    --checkpoint ./outputs/model_best.pth \
    --data_dir ./data \
    --output_dir ./results

# Run inference on single image
python inference.py \
    --checkpoint ./outputs/model_best.pth \
    --image ./test_image.jpg \
    --output ./result.jpg
```

## 📊 Key Innovations

### 1. Fused Post-Processing Strategy

Combines heatmap peak detection with direct coordinate regression:

```
Final Coordinates = α × Heatmap_Coords + (1-α) × Regression_Coords
```

- **Benefits**: Enhanced scale robustness for diverse body sizes
- **Handles**: Subtle motions and low-resolution features

### 2. Morphology-Aware Shape Constraint Loss

```python
L_morph = λ × ||Var(P) - Var(GT)||²
```

Where:
- `Var(P)`: Spatial variance of predicted heatmap
- `Var(GT)`: Spatial variance of ground-truth heatmap
- `λ`: Weight hyperparameter (default: 0.1)

**Purpose**: Enforces distribution consistency, reducing Gaussian bias errors

### 3. Coordinate Correspondence Refinement

Joint optimization that:
- Reduces pixel downsampling artifacts
- Minimizes coordinate deviations
- Improves localization under poor feature diversity

## 📈 Performance

| Method | AP | AP@0.5 | AP@0.75 | PCK@0.2 |
|--------|------|--------|---------|---------|
| Baseline HRNet | 88.3 | 96.5 | 92.1 | 91.2 |
| + Fused Head | 92.7 | 97.8 | 94.3 | 94.5 |
| + Morph Loss | 94.1 | 98.2 | 95.6 | 95.8 |
| **Ours (Full)** | **95.4** | **98.9** | **96.8** | **96.7** |

*Evaluated on Infant-Skeleton-V2 dataset from Jiaxing Maternity and Child Health Hospital*

## ⚙️ Configuration

Key parameters in `configs/default.yaml`:

```yaml
MODEL:
  NAME: 'pose_hrnet'
  NUM_JOINTS: 13
  IMAGE_SIZE: [256, 256]
  HEATMAP_SIZE: [64, 64]
  FUSED_HEAD: True
  
LOSS:
  MORPH_WEIGHT: 0.1
  MORPH_LAMBDA: 1.0
  REG_WEIGHT: 0.5
  
TRAIN:
  BATCH_SIZE: 32
  LR: 0.001
  EPOCHS: 200
  VAL_INTERVAL: 5
```

## 🔧 Advanced Usage

### Custom Dataset Conversion

```bash
python tools/convert_to_coco.py \
    --input_dir ./raw_annotations \
    --output_file ./data/annotations/train.json \
    --keypoint_names nose left_eye right_eye ...
```

### Visualize Predictions

```bash
python utils/visualization.py \
    --predictions ./results/predictions.json \
    --images_dir ./data/images/val \
    --output_dir ./visualizations
```

### Analyze Dataset

```bash
python tools/analyze_dataset.py \
    --ann_file ./data/annotations/train.json \
    --output_report ./analysis_report.pdf
```

## 📝 Citation

```bibtex
@article{preterm_infant_pose_2025,
  title={Non-Contact Vision-Based Pose Estimation for Preterm Infant Limb Movement Recognition},
  author={Your Name},
  journal={Journal Name},
  year={2025},
  note={AP: 95.4\% on Infant-Skeleton-V2}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Data collected at Jiaxing Maternity and Child Health Hospital
- Built on [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- COCO evaluation tools from [cocoapi](https://github.com/cocodataset/cocoapi)

## 📧 Contact

For questions or collaboration:
- Email: baoxf96@163.com
- Issues: [GitHub Issues](https://github.com/MarkJhonBao/preterm-infant-pose/issues)

---

**Note**: This system is designed for research and clinical decision support. Always consult healthcare professionals for medical decisions regarding preterm infant care.
