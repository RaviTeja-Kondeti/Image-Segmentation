# UNet Semantic Segmentation with VGG16

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

Production-ready implementation of **UNet architecture** with **VGG16 encoder** for pixel-level semantic image segmentation. This implementation features a custom upsampling path with batch normalization and Feature Pyramid Network (FPN)-style skip connections for enhanced localization.

### Key Features

- **Transfer Learning**: Pre-trained VGG16 backbone for robust feature extraction
- **Custom Upsampling Path**: Transpose convolutions with batch normalization
- **FPN-Style Architecture**: Multi-scale feature fusion via skip connections
- **Production Ready**: Clean, modular code with best practices
- **Flexible**: Easy to adapt for different segmentation tasks

## Architecture

```
Input Image (H x W x 3)
        ↓
   VGG16 Encoder (Downsampling)
   ├── Conv Block 1 → Skip Connection 1
   ├── Conv Block 2 → Skip Connection 2  
   ├── Conv Block 3 → Skip Connection 3
   ├── Conv Block 4 → Skip Connection 4
   └── Conv Block 5 (Bottleneck)
        ↓
   Custom Decoder (Upsampling)
   ├── TransposeConv + BatchNorm + Skip 4
   ├── TransposeConv + BatchNorm + Skip 3
   ├── TransposeConv + BatchNorm + Skip 2
   └── TransposeConv + BatchNorm + Skip 1
        ↓
Output Segmentation Map (H x W x num_classes)
```

## Technical Highlights

- **Encoder**: VGG16 pre-trained on ImageNet
- **Decoder**: Custom upsampling with transpose convolutions
- **Skip Connections**: Concatenation of encoder features at multiple scales
- **Normalization**: Batch normalization after each upsampling block
- **Output**: Same spatial dimensions as input for pixel-perfect segmentation

## Requirements

```bash
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/RaviTeja-Kondeti/Image-Segmentation.git
cd Image-Segmentation

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
import torch
from model import UNetVGG16

# Initialize model
model = UNetVGG16(num_classes=21)
model.eval()

# Load input image (preprocessed)
input_tensor = torch.randn(1, 3, 512, 512)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    predictions = torch.argmax(output, dim=1)
```

## Model Architecture Details

### Downsampling Path (VGG16)
1. **Block 1**: 2 × Conv(64) + MaxPool
2. **Block 2**: 2 × Conv(128) + MaxPool
3. **Block 3**: 3 × Conv(256) + MaxPool
4. **Block 4**: 3 × Conv(512) + MaxPool
5. **Block 5**: 3 × Conv(512) + MaxPool

### Upsampling Path (Custom)
1. **Up Block 1**: TransposeConv(512→512) + BatchNorm + Concat(Block4)
2. **Up Block 2**: TransposeConv(512→256) + BatchNorm + Concat(Block3)
3. **Up Block 3**: TransposeConv(256→128) + BatchNorm + Concat(Block2)
4. **Up Block 4**: TransposeConv(128→64) + BatchNorm + Concat(Block1)
5. **Output**: Conv(64→num_classes)

## Performance Considerations

- **Input Size**: Flexible, but power of 2 recommended (256×256, 512×512)
- **Memory**: ~45M parameters with VGG16 encoder
- **Speed**: Real-time inference on modern GPUs
- **Accuracy**: Competitive with state-of-the-art segmentation models

## Applications

- Medical image segmentation
- Autonomous driving (road scene understanding)
- Satellite imagery analysis
- Object instance segmentation
- Industrial defect detection

## Project Structure

```
Image-Segmentation/
├── model.py                 # UNet-VGG16 architecture
├── train.py                 # Training script
├── inference.py             # Inference utilities
├── data/                    # Dataset handlers
├── utils/                   # Helper functions
├── checkpoints/             # Saved models
└── README.md
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{unet_vgg16_segmentation,
  author = {Kondeti, Ravi Teja},
  title = {UNet Semantic Segmentation with VGG16},
  year = {2025},
  url = {https://github.com/RaviTeja-Kondeti/Image-Segmentation}
}
```

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://arxiv.org/abs/1409.1556)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Ravi Teja Kondeti**
- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

---

⭐ **Star this repository** if you find it helpful!
