# ULS-MSA: Ultra-Lightweight Skin Lesion Segmentation for Practical Deployment in Resource-Limited Clinical Settings

This is the official implementation of the **ULS-MSA** (ULtra-Lightweight Segmentation with Multi-Scale Edge-Aware Subspace Attention) model as proposed in our paper submitted to *Neural Networks*].

📄 **Paper**:  [Link to be added]  
🚀 **Key Highlights**: 0.24M Parameters | 2.6 GFLOPs | 0.0055s Inference Time 

ULS-MSA is specifically engineered for resource-constrained environments—such as rural clinics, primary care offices, and mobile health (mHealth) applications—where high-end GPU servers are unavai[...]

---

## 🏗️ Model Architecture

The network employs a unique **"division of labor"** strategy to achieve high precision without computational bloat:
* **Boundary Detection Module (BDM)**: Utilizes fixed-weight Gaussian and Laplacian kernels to explicitly capture high-frequency edge details, avoiding the prohibitive costs of fully learnable edge mo[...]
* **Squeeze-Edge Attention Lightweight (SEAL) Block**: Recalibrates semantic features efficiently using enhanced depthwise separable convolutions (EDSC) and channel attention.
* **Optimized for Deployment**: With only **0.24 million parameters**, ULS-MSA is over 4.7x smaller than existing lightweight models like LightMUNet. 

---

## 📊 Performance Results

ULS-MSA consistently outperforms state-of-the-art lightweight models across three challenging public benchmark datasets.

### Quantitative Comparison (mIoU & F1-Score)
| Dataset | mIoU | F1-Score | Precision | Recall | Params |
| :--- | :---: | :---:  | :---: | :---:  | :---: |
| **ISIC-2017** | **0.7989** | **0.7978** | 0.9227 | 0.7499 | 0.24M |
| **ISIC-2018** | **0.8408** | **0.8693** | 0.8772 | 0.8936 | 0.24M |
| **PH2** | **0.9040** | **0.9379** | 0.9443 | 0.9355 | 0.24M |

*Data compiled from Tables 2 and 3 of the manuscript.*

---

## 🚀 Quick Start

### 1. Requirements
The experiments were implemented in **PyTorch** and run on a workstation with an Intel Core i5-8500 and an NVIDIA GeForce GTX 1070.

* python 3.7.5+
* pytorch 2.2.0
* opencv 4.9.0
* numpy 1.26.4
* scipy 1.11.4
* matplotlib 3.8.0

### 2. Installation
```bash
git clone https://github.com/razanalharith/ULS-MSA. git
cd ULS-MSA
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

We employ three publicly available benchmarks for skin lesion segmentation: 

### Supported Datasets

1. **ISIC-2017 Segmentation Dataset**
   - 2,000 dermoscopic images with ground truth masks
   - [Download from ISIC Archive](https://challenge.isic-archive.com/data/)
   
2. **ISIC-2018 Task 1: Lesion Boundary Segmentation**
   - 3,694 dermoscopic images with pixel-level annotations
   - [Download from ISIC Archive](https://challenge.isic-archive.com/data/)
   
3. **PH2 Dataset**
   - 200 dermoscopic images from Pedro Hispano Hospital
   - [Download Link](https://www.fc.up.pt/addi/ph2%20database.html)

### Directory Structure

After downloading, organize the datasets as follows:

```
ULS-MSA/
├── data/
│   ├── ISIC2017/
│   │   ├── images/
│   │   └── masks/
│   ├── ISIC2018/
│   │   ├── images/
│   │   └── masks/
│   └── PH2/
│       ├── images/
│       └── masks/
├── models/
├── utils/
└── train.py
```

---

## 🎯 Training

To train ULS-MSA on a specific dataset: 

```bash
# Train on ISIC-2017
python train.py --dataset ISIC2017 --epochs 100 --batch_size 16

# Train on ISIC-2018
python train.py --dataset ISIC2018 --epochs 100 --batch_size 16

# Train on PH2
python train.py --dataset PH2 --epochs 100 --batch_size 8
```

### Training Options
- `--dataset`: Dataset name (ISIC2017, ISIC2018, PH2)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--save_path`: Path to save trained models

---

## 🧪 Evaluation

To evaluate a trained model:

```bash
python evaluate.py --dataset ISIC2017 --model_path checkpoints/best_model.pth
```

This will compute:
- Mean Intersection over Union (mIoU)
- F1-Score (Dice Coefficient)
- Precision
- Recall
- Inference time per image

---

## 📝 Citation

If you find this work useful for your research, please cite our manuscript:

```bibtex
@article{alharith2025uls,
  title={Ultra-Lightweight Skin Lesion Segmentation for Practical Deployment in Resource-Limited Clinical Settings},
  author={Razan Alharith, Jiashu Zhang, Ashraf Osman Ibrahim and Zaid Al-Huda},
  journal={Neural Networks},
  year={2025},
  note={Under Review}
}
```

---

## ✉️ Contact

**Razan Alharith**  
Southwest Jiaotong University  
📧 Email: razanalharith@my.swjtu.edu.cn

