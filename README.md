# ULS-MSA: Ultra-Lightweight Skin Lesion Segmentation for Practical Deployment in Resource-Limited Clinical Settings

This is the official implementation of the **ULS-MSA** (ULtra-Lightweight Segmentation with Multi-Scale Edge-Aware Subspace Attention) model as proposed in our paper submitted to *Neural Networks*].

📄 **Paper**: [Link to be added]  
🚀 **Key Highlights**: 0.24M Parameters | 2.6 GFLOPs | 0.0055s Inference Time 

ULS-MSA is specifically engineered for resource-constrained environments—such as rural clinics, primary care offices, and mobile health (mHealth) applications—where high-end GPU servers are unavailable and data privacy is paramount.

---

## 🏗️ Model Architecture

The network employs a unique **"division of labor"** strategy to achieve high precision without computational bloat:
* **Boundary Detection Module (BDM)**: Utilizes fixed-weight Gaussian and Laplacian kernels to explicitly capture high-frequency edge details, avoiding the prohibitive costs of fully learnable edge modules.
* **Squeeze-Edge Attention Lightweight (SEAL) Block**: Recalibrates semantic features efficiently using enhanced depthwise separable convolutions (EDSC) and channel attention.
* **Optimized for Deployment**: With only **0.24 million parameters**, ULS-MSA is over 4.7x smaller than existing lightweight models like LightMUNet.



---

## 📊 Performance Results

ULS-MSA consistently outperforms state-of-the-art lightweight models across three challenging public benchmark datasets.

### Quantitative Comparison (mIoU & F1-Score)
| Dataset | mIoU | F1-Score | Precision | Recall | Params |
| :--- | :---: | :---: | :---: | :---: | :---: |
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
git clone [https://github.com/razanalharith/ULS-MSA.git](https://github.com/razanalharith/ULS-MSA.git)
cd ULS-MSA
pip install -r requirements.txt
