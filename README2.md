# Human Pose Estimation

A PyTorch implementation of top-down human pose estimation based on HRFormer (High-Resolution Transformer) architecture.

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

## Model Architecture

### HRFormer Backbone (Default)
- High-Resolution Transformer architecture
- Window-based Multi-head Self-Attention (W-MSA)
- Relative Position Encoding (RPE)
- Multi-resolution parallel branches with feature fusion
- Drop Path regularization

### HRNet Backbone (Optional)
- Multi-resolution parallel branches
- High-resolution representations throughout
- Multi-scale feature fusion

### Heatmap Head
- Direct 1x1 convolution from backbone features
- Gaussian heatmap supervision
- Sub-pixel coordinate decoding

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
