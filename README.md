# Morphology-Aware Pose Estimation for Preterm Infant Monitoring

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Morphology-Aware Pose Estimation with Gaussian Bias Mitigation for Robust Preterm Infant Movements Monitoring"**.

<div align="center">
  <video src="test.mp4" width="800" controls></video>
  <p><i>Demo: Real-time preterm infant pose estimation in NICU environment</i></p>
</div>

## 👥 Authors

**Xianfu Bao**¹ · **Hyunsoo Shin**² · **Hyunho Hwang**2 · **Huafei Huang**2 · **Peng Lin**2 · **Jiuwen Cao**¹* · **Sungon Lee**²*

¹ Artificial Intelligent Institute, Hangzhou Dianzi University, China  
² Department of Electrical and Electronic Engineering, Hanyang University, South Korea  
³ Division of Neonatology, Jiaxing Maternity and Child Health Care Hospital, China  
⁴ Machine Learning and I-health International Cooperation Base of Zhejiang Province, China  
⁵ Department of Robotics, Hanyang University, South Korea

*Corresponding authors

## 📄 Abstract

Preterm infants are at higher risk for movement dysfunction and neuro-developmental disorders due to organ developmental immaturity and nervous system issues in neonatal intensive care units (NICUs). Objective preterm infant limb movement recognition (PI-LMR) is considered essential for timely clinical care and disease screening. However, current Gaussian heatmap-based human pose estimation algorithms are affected by pixel downsampling and peak drift of Gaussian bias, resulting in detection curve fluctuations.

This work proposes a non-contact vision-based pose estimation framework combining fused head and morphological loss to monitor premature infants' pose movements continuously. To address unstable limb movements and indistinct postures exhibited by preterm infants, which cause movement trajectories to be highly affected by noise interference, we improve the top-down paradigm to eliminate coordinate deviations caused by feature map downsampling and Gaussian bias errors.

**Key innovations include:**
- A post-processing fusion strategy combining heatmap-based and regression-based representations for enhanced scale robustness
- A morphology-aware shape constraint loss enforcing distribution consistency by penalizing spatial variance discrepancies
- A coordinate correspondence refinement mechanism jointly optimizing heatmap localization and regressed keypoints

Extensive experiments demonstrate accurate and robust pose estimation for NICUs, enabling early abnormal posture screening with **95.4% accuracy** validated on Infant-Skeleton-V2 data.

## 📋 Overview

This repository presents a non-contact vision-based framework for accurate and robust pose estimation of preterm infants in Neonatal Intensive Care Units (NICUs). Our method addresses critical challenges in preterm infant limb movement recognition (PI-LMR) through novel approaches to Gaussian bias mitigation and morphology-aware optimization.

### Key Features

- **🎯 Gaussian Bias Mitigation**: Eliminates coordinate deviations caused by feature map downsampling and Gaussian peak shifts
- **🔄 Dual-Branch Architecture**: Combines heatmap-based global spatial perception with regression-based local refinement
- **📐 Morphology-Aware Loss**: Enforces distribution consistency through spatial variance constraints
- **🎨 Coordinate Refinement**: Sub-pixel precision through soft-argmax with local Gaussian refinement
- **⚕️ Clinical Validation**: Achieved 95.4% accuracy on Infant-Skeleton-V2 dataset

## 🏗️ Architecture

Our framework consists of three main components:

1. **Regression-Assisted Feature Fusion Head**
   - Integrates Gaussian heatmap representation with continuous coordinate regression
   - Dual-supervision framework for improved localization stability

2. **Morphological Correlation Loss**
   - Joint optimization of heatmap variance constraints
   - Decouples receptive field overlap and corrects Gaussian bias

3. **Continuous Coordinate Regression**
   - Combines soft-argmax with local Gaussian refinement
   - Achieves sub-pixel precision for enhanced continuity

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (for GPU support)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/preterm-infant-pose-estimation.git
cd preterm-infant-pose-estimation

# Create virtual environment
conda create -n infant_pose python=3.8
conda activate infant_pose

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

```bash
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.4.0
tensorboard>=2.7.0
pycocotools>=2.0.0
```

## 📊 Dataset

The model is validated on **Infant-Skeleton-V2** dataset collected at Jiaxing Maternity and Child Health Hospital.

### Data Structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### Data Preparation

```bash
# Download and prepare the dataset
python tools/prepare_data.py --data-dir ./data

# Verify dataset integrity
python tools/verify_data.py --data-dir ./data
```

## 🏋️ Training

### Quick Start

```bash
# Train with default configuration
python train.py --config configs/default.yaml

# Train with custom settings
python train.py \
    --config configs/custom.yaml \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001
```

### Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  backbone: hrnet_w32
  num_keypoints: 17
  heatmap_size: [64, 64]
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  
loss:
  heatmap_weight: 1.0
  regression_weight: 0.5
  morphology_weight: 0.3
```

## 🔬 Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --data-dir ./data/test

# Generate visualization
python visualize.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input-dir ./data/test/images \
    --output-dir ./results
```

## 📈 Results

| Method | AP | AP@50 | AP@75 | Accuracy |
|--------|-----|-------|-------|----------|
| Baseline | 82.3 | 94.1 | 87.6 | 89.2% |
| + Fusion Head | 86.7 | 95.8 | 90.4 | 92.1% |
| + Morphology Loss | 89.4 | 96.5 | 92.1 | 93.8% |
| **Ours (Full)** | **92.8** | **97.3** | **94.2** | **95.4%** |

## 🎯 Inference

### Single Image

```python
from models import PoseEstimator
from utils import load_image, visualize_pose

# Load model
model = PoseEstimator(config='configs/default.yaml')
model.load_checkpoint('checkpoints/best_model.pth')

# Inference
image = load_image('path/to/image.jpg')
keypoints, confidence = model.predict(image)

# Visualize
visualize_pose(image, keypoints, save_path='output.jpg')
```

### Video Processing

```bash
python demo/video_demo.py \
    --input video.mp4 \
    --output output.mp4 \
    --checkpoint checkpoints/best_model.pth
```

## 📁 Project Structure

```
preterm-infant-pose-estimation/
├── configs/              # Configuration files
├── data/                 # Dataset directory
├── demo/                 # Demo scripts
├── models/               # Model implementations
│   ├── backbone.py
│   ├── fusion_head.py
│   └── pose_estimator.py
├── losses/               # Loss functions
│   ├── heatmap_loss.py
│   ├── regression_loss.py
│   └── morphology_loss.py
├── utils/                # Utility functions
├── tools/                # Data preparation tools
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt
└── README.md
```

## 🔧 Advanced Usage

### Custom Loss Weights

```python
loss_config = {
    'heatmap_weight': 1.0,
    'regression_weight': 0.5,
    'morphology_weight': 0.3
}
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --config configs/default.yaml
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{bao2025morphology,
  title={Morphology-Aware Pose Estimation with Gaussian Bias Mitigation for Robust Preterm Infant Movements Monitoring},
  author={Bao, Xianfu and Shin, Hyunsoo and Hwang, Hyunho and Huang, Huafei and Lin, Peng and Cao, Jiuwen and Lee, Sungon},
  journal={Neural Networks},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset collection: Jiaxing Maternity and Child Health Hospital
- Funding: Machine Learning and I-health International Cooperation Base of Zhejiang Province
- Infrastructure support: Hangzhou Dianzi University & Hanyang University

## 📧 Contact

- **Jiuwen Cao** (Corresponding Author): jwcao@hdu.edu.cn
- **Sungon Lee** (Corresponding Author): sungon@hanyang.ac.kr
- **Xianfu Bao**: baoxf96@163.com

## 🔗 Links

- [Paper](https://arxiv.org/) (Coming soon)
- [Project Page](https://yourprojectpage.com)
- [Demo Video](https://youtu.be/)

## ⚠️ Disclaimer

This tool is intended for research purposes only. Clinical decisions should be made by qualified healthcare professionals. Always consult with medical experts for diagnosis and treatment.

---

**Note**: This is an active research project. Updates and improvements are ongoing.
