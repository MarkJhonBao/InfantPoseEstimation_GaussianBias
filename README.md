# Robust Pose Estimation for Preterm Infant Limb Movement Recognition

[![Paper](https://img.shields.io/badge/Paper-Neural_Networks-blue)](https://doi.org/xxx)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

Official implementation of **"Robust Pose Estimation via Regression Correction and Gaussian Alignment for Preterm Infant Limb Movement Recognition"** (Neural Networks 2026).

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

## 📋 Abstract

Preterm infant pose estimation in neonatal intensive care units (NICUs) is critical for early screening of neurodevelopmental disorders. This work proposes a dual-branch framework that fuses heatmap and regression representations under morphology-aware constraints. Our method addresses three key challenges: **single-frame-dependent keypoint drift**, **discretization-induced localization errors**, and **weak robustness to subtle limb dynamics**.

Our PI-LMR algorithm achieves an accuracy of **93.8%** on the Infant-Skeleton-V2 dataset, outperforming state-of-the-art methods by **2.8%**.

## ✨ Highlights

- 🎯 **Regression-Assisted Fusion Head**: Integrates Gaussian heatmaps with continuous coordinate regression to mitigate quantization errors
- 🔧 **VAMC Mechanism**: Variance Alignment and Morphological Constraints for enhanced structural consistency
- 📐 **Shape Constraint Loss (SCL)**: Achieves sub-pixel precision through soft-argmax combined with local Gaussian refinement
- 🏥 **Clinical Application**: Enables early screening for postural abnormalities in NICU environments

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" width="85%">
</p>

The model consists of three main stages:
1. **Backbone Architecture**: Multi-scale feature extraction with reverted residual blocks
2. **Dual-Head Module**: Parallel Gaussian heatmap and coordinate regression branches
3. **Post-processing Stage**: Morphological constraint optimization for robust localization


## 📊 Dataset

### Infant-Skeleton-V2

Our dataset was collected at **Jiaxing Maternity and Child Health Care Hospital** in collaboration with clinical experts.

| Split | Samples | Description |
|-------|---------|-------------|
| Train | - | Training set with augmentation |
| Val | - | Validation set |
| Test | - | Test set for final evaluation |

### Data Preparation

```bash
# Download the dataset (contact authors for access)
# Organize the data as follows:
data/
├── infant_skeleton_v2/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── README.md
```

### Keypoint Definition

The model predicts **K** anatomical keypoints on the infant body, including:
- Head landmarks
- Upper limb joints (shoulder, elbow, wrist)
- Lower limb joints (hip, knee, ankle)
- Torso landmarks

## 🚀 Usage

### Training

```bash
# Single GPU training
python tools/train.py --config configs/infant_skeleton_v2.yaml

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py --config configs/infant_skeleton_v2.yaml
```

### Evaluation

```bash
# Evaluate on test set
python tools/test.py \
    --config configs/infant_skeleton_v2.yaml \
    --checkpoint checkpoints/best_model.pth
```

### Inference Demo

```bash
# Run inference on a single image
python tools/demo.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output results/

# Run inference on video
python tools/demo.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pth \
    --output results/
```

## 📈 Results

### Performance on Infant-Skeleton-V2

| Method | Accuracy | PCK@0.2 | AUC | FPS |
|--------|----------|---------|-----|-----|
| SimpleBaseline | 87.2 | - | - | - |
| HRNet | 89.5 | - | - | - |
| ViTPose | 91.0 | - | - | - |
| **Ours** | **93.8** | - | - | - |

### Ablation Study

| Component | Accuracy | Δ |
|-----------|----------|---|
| Baseline | 89.5 | - |
| + Regression Head | 91.2 | +1.7 |
| + VAMC | 92.6 | +1.4 |
| + SCL | **93.8** | +1.2 |

## 🔬 Method Details

### Variance Alignment and Morphological Constraints (VAMC)

The VAMC mechanism enforces data distribution consistency by penalizing spatial variance discrepancies across the coordinate regression error gradient:

```
L_VAMC = λ₁ · L_variance + λ₂ · L_morphology
```

### Shape Constraint Loss (SCL)

The SCL combines soft-argmax with local Gaussian refinement to achieve sub-pixel precision:

```
L_SCL = ||p_pred - p_gt||₂ + α · L_shape
```



## Project Structure

```
pose_estimation/
├── configs/
│   ├── __init__.py
│   └── config.py           # Configuration dataclasses
├── datasets/
│   ├── __init__.py
│   ├── coco_dataset.py     # COCO dataset implementation
│   └── transforms.py       # Data augmentation transforms
├── models/
│   ├── __init__.py
│   ├── hrnet.py            # HRNet backbone (optional)
│   ├── hrformer.py         # HRFormer backbone (Transformer-based)
│   └── pose_estimator.py   # Complete pose estimator
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics (OKS, AP)
│   └── visualization.py    # Visualization utilities
├── train.py                # Training script
├── validate.py             # Validation script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd pose_estimation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### COCO Dataset

1. Download COCO 2017 dataset:
   - [train2017](http://images.cocodataset.org/zips/train2017.zip)
   - [val2017](http://images.cocodataset.org/zips/val2017.zip)
   - [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

2. Organize the data:
```
data/
└── coco/
    ├── annotations/
    │   ├── person_keypoints_train2017.json
    │   └── person_keypoints_val2017.json
    ├── train2017/
    │   ├── 000000000001.jpg
    │   └── ...
    └── val2017/
        ├── 000000000139.jpg
        └── ...
```

## Usage

### Training

```bash
# Basic training
python train.py --data_root data/coco/

# Training with custom parameters
python train.py \
    --data_root data/coco/ \
    --batch_size 32 \
    --epochs 210 \
    --lr 5e-4

# Resume training from checkpoint
python train.py \
    --data_root data/coco/ \
    --resume checkpoints/latest.pth
```

### Validation

```bash
# Validate with flip test
python validate.py \
    --checkpoint checkpoints/best.pth \
    --data_root data/coco/

# Validate without flip test
python validate.py \
    --checkpoint checkpoints/best.pth \
    --data_root data/coco/ \
    --no_flip
```

### Inference

```bash
# Single image inference
python inference.py \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best.pth \
    --output result.jpg

# Batch inference on directory
python inference.py \
    --input path/to/images/ \
    --checkpoint checkpoints/best.pth \
    --output path/to/results/

# With specific bounding box
python inference.py \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best.pth \
    --bbox 100 100 300 400 \
    --verbose
```

## Configuration

Configuration is managed through dataclasses in `configs/config.py`:

```python
from configs import get_config

cfg = get_config()

# Modify settings
cfg.data.input_size = (256, 192)
cfg.train.batch_size = 32
cfg.train.lr = 5e-4
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | (192, 256) | Input image size (W, H) |
| `heatmap_size` | (48, 64) | Heatmap size (W, H) |
| `num_keypoints` | 17 | Number of keypoints |
| `batch_size` | 32 | Training batch size |
| `lr` | 5e-4 | Learning rate |
| `max_epochs` | 210 | Maximum training epochs |
| `sigma` | 2.0 | Gaussian sigma for heatmap |
| `head_type` | 'fusion' | Head type: 'heatmap' or 'fusion' |
| `heatmap_loss_weight` | 1.0 | Weight for heatmap loss |
| `offset_loss_weight` | 1.0 | Weight for offset regression loss |
| `variance_loss_weight` | 0.1 | Weight for variance alignment loss |


```

**Key Features:**
- **Sub-pixel Refinement**: Global Soft-Argmax + Local Gaussian fitting
- **Offset Regression**: Corrects quantization error (ε_q ≤ √2/2 × s)
- **Variance Alignment**: Constrains predicted heatmap spread

### Multi-Component Loss Function
```
L_total = λ₁·L_heatmap + λ₂·L_offset + λ₃·L_peak 
        + λ₄·L_variance + λ₅·L_overlap + λ₆·L_shape
```

| Loss | Description | Default Weight |
|------|-------------|----------------|
| L_heatmap | Heatmap MSE loss | 1.0 |
| L_offset | Offset regression (SmoothL1) | 1.0 |
| L_peak | Peak localization (L2) | 0.5 |
| L_variance | Variance alignment | 0.1 |
| L_overlap | Spatial overlap regularization | 0.05 |
| L_shape | Distribution shape (entropy) | 0.05 |

### Gaussian Distribution Constraints
- **Variance Alignment**: σ_pred → σ_gt (target sigma = 2.0)
- **Spatial Overlap**: Prevents adjacent keypoint ambiguity
- **Shape Constraint**: Encourages unimodal Gaussian distribution

### HRNet Backbone (Optional)
- Multi-resolution parallel branches
- High-resolution representations throughout
- Multi-scale feature fusion

## Evaluation Metrics

Following COCO keypoint evaluation protocol:
- **AP**: Average Precision at OKS = 0.50:0.05:0.95
- **AP50**: AP at OKS = 0.50
- **AP75**: AP at OKS = 0.75
- **AP_M**: AP for medium objects
- **AP_L**: AP for large objects
- **AR**: Average Recall

## Expected Results

| Model | Input Size | AP | AP50 | AP75 |
|-------|------------|-----|------|------|
| HRFormer-Base | 256x192 | 75.6 | 90.8 | 82.8 |
| HRFormer-Base | 384x288 | 77.2 | 91.0 | 83.6 |
| HRNet-W32 | 256x192 | 74.4 | 90.5 | 81.9 |
| HRNet-W48 | 384x288 | 76.3 | 90.8 | 82.9 |

## Training Tips

1. **Learning Rate**: Use warmup for first 5 epochs
2. **Data Augmentation**: Enable random flip, rotation, and half-body
3. **Batch Size**: Larger batch size improves stability
4. **Mixed Precision**: Enable FP16 for faster training

## Citation

```bibtex
@inproceedings{yuan2021hrformer,
  title={HRFormer: High-Resolution Transformer for Dense Prediction},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}
```

## License

This project is released under the Apache 2.0 License.

## Acknowledgements

- [MMPose](https://github.com/open-mmlab/mmpose)
- [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
