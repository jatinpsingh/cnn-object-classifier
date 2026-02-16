# Image Classification Project

A CNN-based object classification system capable of classifying 39 different object classes. This project supports training custom CNNs as well as fine-tuning state-of-the-art transfer learning models (EfficientNet, ConvNeXt, MobileNet).

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Data Augmentation](#2-data-augmentation)
  - [3. Training](#3-training)
  - [4. Evaluation & Inference](#4-evaluation--inference)
- [Available Models](#available-models)

## 🚀 Project Overview

- **Goal:** Classify images into one of 39 categories (e.g., OBJ_001, OBJ_002, ...).
- **Framework:** TensorFlow/Keras.
- **Features:** 
  - Config-driven training (YAML).
  - Offline data augmentation pipeline (Albumentations).
  - Custom CNN and Transfer Learning support.
  - Automated train/val/test splitting.

## 🛠 Environment Setup

This project uses a Conda environment named `image_classification`.

1. **Create and Activate Environment:**
   ```bash
   conda create -n image_classification python=3.10
   conda activate image_classification
   ```

2. **Install Dependencies:**
   Ensure you have the necessary libraries. Using a GPU-enabled TensorFlow is highly recommended.
   ```bash
   conda install tensorflow-gpu albumentations opencv pyyaml tqdm numpy
   # Or via pip if conda packages are missing
   # pip install tensorflow albumentations opencv-python pyyaml tqdm numpy
   ```

## 📂 Project Structure

```
.
├── configs/                     # YAML configuration files for training
├── data/                        # Raw dataset (39 class subfolders)
├── augmented_data/              # Generated augmented images
├── outputs/                     # Training artifacts (models, logs)
├── notebooks/                   # Jupyter notebooks for inference/analysis
├── src/                         # Source code
│   ├── augment.py               # Offline data augmentation script
│   ├── dataset.py               # TensorFlow dataset pipeline
│   ├── models.py                # Model architectures and factory
│   ├── prepare_data.py          # Data discovery and split manifest creation
│   ├── train.py                 # Main training script
│   └── evaluate.py              # Model evaluation script
├── split_manifest.json          # Train/Val/Test splits (Original)
└── split_manifest_augmented.json # Train/Val/Test splits (with Augmentation)
```

## 🔄 Workflow

### 1. Data Preparation
Scans the `data/` directory and creates a JSON manifest defining train/val/test splits.

```bash
python src/prepare_data.py --data_dir data --output split_manifest.json
```

### 2. Data Augmentation
Performs offline augmentation on the training split. This creates a new directory `augmented_data/` and a new manifest.

```bash
python src/augment.py --manifest split_manifest.json --output_dir augmented_data
```

### 3. Training
Train a model using a specific configuration file. The script uses `split_manifest_augmented.json` by default if configured in the YAML.

**Basic Usage:**
```bash
python src/train.py --config configs/custom_cnn.yaml
```

**Override Hyperparameters:**
You can override config values directly from the CLI:
```bash
python src/train.py --config configs/efficientnetv2s.yaml --epochs 50 --learning_rate 0.0001
```

### 4. Evaluation & Inference
- **Evaluation:** Run `src/evaluate.py` (if available/configured) to test model performance.
- **Inference:** Check `notebooks/single_object_inference.ipynb` for examples on how to run inference on single images.

## 🧠 Available Models

Configuration files can be found in `configs/`:

| Config File | Model Type | Description |
| :--- | :--- | :--- |
| `custom_cnn.yaml` | Custom CNN | A simple custom CNN architecture. |
| `efficientnetv2s.yaml` | EfficientNetV2S | Transfer learning using EfficientNetV2 Small (ImageNet weights). |
| `convnexttiny.yaml` | ConvNeXtTiny | Transfer learning using ConvNeXt Tiny. |
| `mobilenetv2.yaml` | MobileNetV2 | Lightweight model suitable for mobile. |
| `mobilenetv3large.yaml`| MobileNetV3Large | Optimized MobileNet V3 Large. |
| `mobilenetv3small.yaml`| MobileNetV3Small | Optimized MobileNet V3 Small. |

## 📝 Notes
- **Path Resolution:** Scripts in `src/` automatically resolve the project root, so they can be run from anywhere in the project tree.
- **Normalization:** 
  - `src/dataset.py` loads images as `float32` [0-255].
  - `src/models.py` handles normalization (e.g., `Rescaling(1./255)` or model-specific preprocessing).
