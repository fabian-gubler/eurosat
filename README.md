# EuroSAT Project: Satellite Image Classification

### :books: **Final Presentation**:  
**[Eurosat Coding Challenge Report](https://raw.githubusercontent.com/fabian-gubler/eurosat/main/Eurosat_FinalPresentation.pdf)** – Click the link for a final presentation of our results.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation & Usage](#installation--usage)
- [Key Learnings](#key-learnings)

## Overview

The EuroSAT project focuses on the classification of satellite images using various neural network architectures, including Transfer Learning and Self-Supervised Learning approaches. The main goal was to mitigate domain shift between different satellite image levels (L1C to L2A) by leveraging techniques like correlation alignment, multi-spectral data augmentation, and fine-tuning pre-trained models like ResNet-50.

The model was trained on EuroSAT data, achieving significant improvements in test accuracy using pseudo-labeling techniques and domain adaptation strategies.

## Repository Structure
```bash
models/            # Contains trained models (.h5, .pt)
predictions/       # Stores predictions for test data in CSV format
src/               # Source code for different ResNet models and training scripts
```

## Installation & Usage

### Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- PyTorch (for some models)
- AWS credentials if using AWS compute instances

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/fabian-gubler/eurosat.git
    cd eurosat-project
    ```

2. Install dependencies:
    ```bash
    pip install -r src/requirements.txt
    ```

## Key Learnings

- **Domain Adaptation**: Handling domain shifts was one of the key challenges. We employed techniques like correlation alignment (CORAL) and pseudo-labeling to generalize models trained on one domain to another.
- **Transfer Learning**: Fine-tuning pre-trained models like ResNet-50 significantly reduced training time while improving model performance on multi-spectral data.
- **Self-Supervised Learning**: Leveraging self-supervised learning helped in making the model more robust when facing unlabeled data, especially in scenarios with limited training examples.
