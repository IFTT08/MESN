# MESN: Multi-Exit Stitched Network

This repository provides the complete implementation of the experiments
for the paper:

**Representation-Similarity Network Stitching for Efficient Parameter
Reduction in Multi-Exit Networks**

The proposed method constructs a **Multi-Exit Stitched Network (MESN)**
by identifying representation-similar feature layers across intermediate
classifiers and connecting them through lightweight stitching layers.
This enables redundant feature reduction layers to be removed and shared
across multiple exits, thereby reducing parameter redundancy while
preserving the predictive capability of multi-exit networks.

The repository contains the implementation of the baseline Multi-Exit
Network (MEN), MESN construction and training, the proposed IDD training
strategy, adaptive inference experiments, and comparative experiments
with different compression methods.

------------------------------------------------------------------------

## Overview

Multi-Exit Networks (MENs) introduce intermediate classifiers at
different depths of a backbone network, allowing predictions to be
generated before the final layer. Although intermediate classifiers
provide adaptive inference capability, independently maintaining feature
reduction layers for multiple exits introduces considerable parameter
redundancy.

This work addresses this problem through **network stitching**. By
connecting representation-similar feature representations from different
exits, feature reduction layers can be progressively reused across
multiple intermediate classifiers.

The proposed **MESN** consists of:

-   Representation-similarity-based network stitching
-   Lightweight (1`\times1`{=tex}) convolution stitching layers
-   Data-driven initialization
-   Knowledge distillation
-   Stitching direction selection
-   Progressive reuse of feature reduction layers

This repository provides the implementation required to reproduce the
main experiments reported in the paper.

------------------------------------------------------------------------

## Main Components

### 1. Baseline Multi-Exit Network

The repository provides training and testing implementations for the
original Multi-Exit Network.

-   `train_menet.py` --- training of the baseline MEN
-   `test_menet.py` --- testing of the baseline MEN

The baseline MEN is used as the reference network for evaluating
parameter reduction and classification performance after network
stitching.

### 2. Multi-Exit Stitched Network

The proposed MESN is constructed by stitching feature representations
between intermediate classifiers.

The implementation includes:

-   MESN construction
-   Stitching layer implementation
-   Feature representation transformation
-   Parameter sharing
-   Training and testing of stitched multi-exit networks

The corresponding scripts are:

-   `train_self_trans.py` --- training of the stitched multi-exit
    network
-   `test_self_trans.py` --- testing of the stitched multi-exit network

### 3. IDD Training Strategy

The proposed IDD training strategy is implemented in:

``` text
helper/loops.py
```

IDD consists of three complementary components:

1.  **Data-driven initialization (Init)**\
    Provides suitable initial parameters for the stitching layer based
    on the feature representations to be connected.

2.  **Knowledge distillation (KD)**\
    Uses the predictive information from the original computational path
    to guide the optimization of the newly constructed stitched path.

3.  **Stitching direction selection (Dir)**\
    Determines how branch networks are progressively connected and how
    feature reduction layers are reused across different exits.

### 4. Adaptive Inference

The repository includes experiments for evaluating the adaptive
inference capability of the proposed MESN.

The corresponding implementation is:

``` bash
python adaptive_inference.py
```

The script includes two adaptive inference scenarios:

-   **Anytime prediction**
-   **Budgeted batch classification**

These experiments evaluate the behavior of different exits under varying
inference requirements and computational budgets.

### 5. Different Compression Methods

The implementation of the experiment **Impact of Different Compression
Methods on Multi-Exit Network Performance** is provided in:

``` bash
python construct_model_svd.py
```

This experiment compares the proposed network stitching strategy with
other parameter compression approaches considered in the paper.

------------------------------------------------------------------------

## Project Structure

``` text
MESN/
│
├── comparators/
│   └──                     # Comparison methods
│
├── dataset/
│   └──                     # Dataset-related files
│
├── helper/
│   └── loops.py            # Training loops, including IDD training
│
├── models/
│   └──                     # Backbone, MEN and MESN models
│
├── utils/
│   └──                     # Utility functions
│
├── README.md
│
├── adaptive_inference.py   # Anytime prediction and budgeted batch classification
│
├── construct_model_svd.py  # Different compression methods
│
├── test_menet.py           # Test baseline MEN
│
├── test_self_trans.py      # Test MESN
│
├── train.sh                # Training commands
│
├── train_menet.py          # Train baseline MEN
│
└── train_self_trans.py     # Train MESN
```

------------------------------------------------------------------------

## Experimental Pipeline

The general experimental workflow is:

``` text
Backbone Network
       │
       ▼
Multi-Exit Network (MEN)
       │
       ├──────────────► Baseline Evaluation
       │
       ▼
Representation Similarity Analysis
       │
       ▼
Network Stitching
       │
       ▼
Multi-Exit Stitched Network (MESN)
       │
       ▼
IDD Training
       │
       ├── Data-driven Initialization
       ├── Knowledge Distillation
       └── Stitching Direction Selection
       │
       ▼
MESN Evaluation
       │
       ├── Classification Performance
       ├── Parameter Reduction
       ├── Exit-wise Performance
       └── Inference Efficiency
       │
       ▼
Adaptive Inference Evaluation
       │
       ├── Anytime Prediction
       └── Budgeted Batch Classification
```

------------------------------------------------------------------------

# Experiments

## 1. Baseline Multi-Exit Network

Train the baseline MEN:

``` bash
python train_menet.py
```

Test the trained MEN:

``` bash
python test_menet.py
```

The baseline results are used as the reference for evaluating the
proposed MESN.

------------------------------------------------------------------------

## 2. Multi-Exit Stitched Network

Train the proposed MESN:

``` bash
python train_self_trans.py
```

Test the trained MESN:

``` bash
python test_self_trans.py
```

The results are used to evaluate the effect of network stitching on
predictive performance and model complexity.

------------------------------------------------------------------------

## 3. IDD Training Strategy

The IDD training strategy is integrated into the training procedure
through:

``` text
helper/loops.py
```

The implementation incorporates:

-   Data-driven initialization
-   Knowledge distillation
-   Stitching direction selection

These components jointly support the optimization of the stitching
layers and the construction of the MESN.

------------------------------------------------------------------------

## 4. Adaptive Inference

Run the adaptive inference experiments:

``` bash
python adaptive_inference.py
```

### Anytime Prediction

Anytime prediction evaluates how prediction performance changes as
additional computation is allowed.

This experiment is used to examine whether MESN preserves the
progressive prediction capability of the original MEN while reducing
parameter redundancy.

### Budgeted Batch Classification

Budgeted batch classification evaluates the utilization of different
exits under a limited computational budget.

This experiment analyzes the trade-off between prediction performance
and computational cost after network stitching.

------------------------------------------------------------------------

## 5. Impact of Different Compression Methods

Run the compression-method comparison:

``` bash
python construct_model_svd.py
```

This corresponds to the experiment:

**Impact of Different Compression Methods on Multi-Exit Network
Performance**

------------------------------------------------------------------------

# Evaluation

The experiments evaluate MEN and MESN from several complementary
perspectives.

### Classification Performance

Classification accuracy is evaluated for different exits to determine
whether network stitching affects the predictive capability of
intermediate classifiers.

### Parameter Reduction

The number of parameters is measured before and after stitching to
quantify the reduction of redundant parameters in intermediate
classifiers.

### Computational Complexity

The inference characteristics of different exits include:

-   FLOPs
-   Inference latency
-   Peak memory consumption

### Adaptive Inference

The adaptive inference experiments evaluate the behavior of different
exits under varying inference requirements and computational budgets.

------------------------------------------------------------------------

# Datasets

The experiments use:

-   CIFAR-10
-   CIFAR-100
-   Tiny-ImageNet

Please prepare the corresponding datasets according to the directory
structure expected by the implementation under:

``` text
dataset/
```

------------------------------------------------------------------------

# Requirements

The implementation is based on **PyTorch**.

A GPU environment with CUDA support is recommended for reproducing the
experiments.

Please refer to the source code and `train.sh` for the corresponding
experimental configuration.

------------------------------------------------------------------------

# Training

The main training scripts are:

``` text
train_menet.py
train_self_trans.py
```

For convenience, the repository also provides:

``` bash
bash train.sh
```

The training configuration can be adjusted according to the selected
dataset, backbone network, number of exits, and experimental setting.

------------------------------------------------------------------------

# Testing

Test the baseline MEN:

``` bash
python test_menet.py
```

Test the proposed MESN:

``` bash
python test_self_trans.py
```

The testing procedures provide exit-wise classification results for
evaluating the predictive performance of the multi-exit networks.

------------------------------------------------------------------------

# Reproducibility

To facilitate independent verification and further research, this
repository provides the implementation required for reproducing the main
experiments in the paper.

The repository includes:

-   Baseline MEN training
-   Baseline MEN testing
-   MESN training
-   MESN testing
-   IDD training
-   Adaptive inference experiments
-   Compression-method comparison
-   Model implementations
-   Supporting utilities and training procedures

------------------------------------------------------------------------

# Relationship to the Paper

  Paper Component                 Implementation
  ------------------------------- --------------------------
  Baseline MEN training           `train_menet.py`
  Baseline MEN testing            `test_menet.py`
  MESN training                   `train_self_trans.py`
  MESN testing                    `test_self_trans.py`
  IDD training strategy           `helper/loops.py`
  Adaptive inference              `adaptive_inference.py`
  Different compression methods   `construct_model_svd.py`
  Model definitions               `models/`
  Supporting utilities            `helper/`, `utils/`
  Comparison methods              `comparators/`
  Dataset-related files           `dataset/`

------------------------------------------------------------------------

# Citation

If you use this code or the proposed MESN in your research, please cite
the corresponding paper:

``` bibtex
@article{MESN,
  title={Representation-Similarity Network Stitching for Efficient Parameter Reduction in Multi-Exit Networks},
  author={Anonymous},
  journal={Applied Soft Computing},
  year={2025}
}
```

Please replace the citation information with the final publication
details after the paper is published.

------------------------------------------------------------------------

# Acknowledgements

We thank the authors of the open-source projects and datasets used in
this work for making their implementations publicly available.
