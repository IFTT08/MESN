# ME-Net: A Unified Experimental Benchmark for Multi-Exit Network Design

## Overview

ME-Net is a unified experimental framework for systematically studying the design factors of Multi-Exit Networks (MENs). Multi-exit networks improve inference efficiency by introducing intermediate classifiers at different depths of a neural network, allowing predictions to be generated before reaching the final layer.

Existing studies often employ different network architectures, classifier structures, training settings, and evaluation protocols, making fair comparison difficult. This repository provides a reproducible benchmark for analyzing the influence of key design factors under controlled experimental conditions.

The framework supports multiple backbone networks, configurable intermediate classifiers, flexible exit placement strategies, varying numbers of exits, and different training methods.

The implementation is based on PyTorch and has been evaluated on CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

---

# Project Structure

```text
ME-Net/
│
├── datasets/
├── helper/
├── models/
├── train_menm.py
├── train_menm8.py
├── test_iconly.py
├── test_menm.py
│
└── README.md
```

---

# Research Objectives

This project investigates the following key design factors in Multi-Exit Networks:

1. Backbone Network Architecture
2. Intermediate Classifier Structure
3. Intermediate Classifier Placement
4. Number of Intermediate Classifiers
5. Training Strategy

All factors can be independently configured and evaluated.

---

# Supported Datasets

## CIFAR-10

CIFAR-10 contains 60,000 color images of size 32×32 distributed among 10 classes.

- Training Images: 50,000
- Test Images: 10,000

Reference:

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.

Website:

https://www.cs.toronto.edu/~kriz/cifar.html

---

## CIFAR-100

CIFAR-100 is similar to CIFAR-10 but contains 100 object categories.

- Training Images: 50,000
- Test Images: 10,000
- Classes: 100

Reference:

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.

Website:

https://www.cs.toronto.edu/~kriz/cifar.html

---

## Tiny-ImageNet

Tiny-ImageNet is a subset of ImageNet containing 200 classes.

- Training Images per Class: 500
- Validation Images per Class: 50
- Test Images per Class: 50
- Image Size: 64×64

Dataset Source:

https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

---

# Requirements

## Hardware

- NVIDIA GPU (recommended)
- CUDA-compatible environment

## Software

The original experiments were conducted using:

- Ubuntu 16.04.5 LTS
- Python 3.7
- PyTorch 1.4.0
- CUDA 11.0

The code should also work with newer PyTorch versions (≥ 1.4.0).

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_repository/ME-Net.git
cd ME-Net
```

Install dependencies:

```bash
pip install torch torchvision numpy scipy tqdm
```

Verify PyTorch installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Code Information

## Backbone Network Training

Train a baseline backbone network without intermediate exits:

```bash
python train_backbone.py
```

Supported backbones include:

- ResNet18
- MobileNetV1
- Other configurable architectures

---

## Multi-Exit Network Training

Train a standard Multi-Exit Network:

```bash
python train_menm.py
```

---

## Multi-Exit Network with Multiple Exits

Train networks containing larger numbers of exits:

```bash
python train_menm8.py
```

This script is mainly used for studying the effect of exit quantity.

---

## Testing

Evaluate trained models:

```bash
python test_menm.py
```

The script reports classification accuracy for all exits and the final classifier.

---

# Intermediate Classifier Configuration

Intermediate classifier structures can be modified in:

```text
models/util/
```

Researchers can customize:

- Classifier depth
- Feature aggregation strategy
- Pooling operations
- Fully connected layers

This enables systematic comparison of different intermediate classifier designs.

---

# Experimental Methodology

The experimental workflow follows these steps:

### Step 1: Train Backbone Networks

Train baseline models without exits.

```bash
python train_backbone.py
```

### Step 2: Insert Intermediate Classifiers

Configure:

- Classifier structure
- Classifier position
- Number of exits

Location:

```text
models/util/
```

### Step 3: Train Multi-Exit Networks

```bash
python train_menm.py
```

or

```bash
python train_menm8.py
```

### Step 4: Evaluate Performance

```bash
python test_menm.py
```

Evaluation metrics include:

- Classification Accuracy
- Exit-wise Accuracy
- Model Complexity
- Computational Overhead

---

# Reproducibility

To ensure fair comparison:

- The same datasets are used across experiments.
- Training hyperparameters remain consistent.
- Only one design factor is modified at a time.
- Multiple backbone networks are evaluated under identical settings.

This allows the isolated study of each design choice.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{ME-Net2025,
  title={A Unified Experimental Benchmark for Multi-Exit Network Design},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

If the paper has been accepted, please replace the above entry with the final publication information.

---

# License

This project is released for academic and research purposes.

You may modify and redistribute the code with proper attribution.

For commercial use, please contact the authors.

---

# Acknowledgements

This repository is built upon the following excellent open-source projects:

### MSDNet

https://github.com/gaohuang/MSDNet

Huang et al., *Multi-Scale Dense Networks for Resource Efficient Image Classification*, ICLR 2018.

### RepDistiller

https://github.com/HobbitLong/RepDistiller

Tian et al., *Contrastive Representation Distillation*, ICLR 2020.

We sincerely thank the authors for making their code publicly available.


