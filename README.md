# Object Classification Pipeline

A CNN-based object classification pipeline capable of classifying 39 different object classes. This project supports training custom CNNs as well as fine-tuning state-of-the-art transfer learning models (EfficientNet, ConvNeXt, MobileNet).

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Data Augmentation](#2-data-augmentation)
  - [3. Training](#3-training)
  - [4. Evaluation & Inference](#4-evaluation--inference)
- [Available Models](#available-models)

## Project Overview

- **Goal:** Classify images into one of 39 categories (e.g., OBJ_001, OBJ_002, ...).
- **Framework:** TensorFlow/Keras.
- **Features:** 
  - Config-driven training (YAML).
  - Offline data augmentation pipeline (Albumentations).
  - Custom CNN and Transfer Learning support.
  - Automated train/val/test splitting.

## Environment Setup

This project uses a Conda environment named `image_classification`.

1. **Create and Activate Environment:**
   ```bash
   conda create -n image_classification python=3.12
   conda activate image_classification
   ```

2. **Install Dependencies:**
   Install the required libraries using `pip` and the provided `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── configs/                     # YAML configuration files for training
├── data/                        # Raw dataset (39 class subfolders)
├── augmented_data/              # Generated augmented images
├── outputs/                     # Training artifacts (models, logs)
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

## Workflow

### 1. Dataset Download
Download the dataset from the following link:
[Google Drive Link](https://drive.google.com/drive/folders/1lQW22uf1tpphMuNlPRoQ8M4smt9w9qLB?usp=drive_link)

Extract/place the dataset folder (containing the 39 class subfolders) into the project directory so that it resides at `data/`.

### 2. Data Preparation
Scans the `data/` directory and creates a JSON manifest defining train/val/test splits.

```bash
python src/prepare_data.py --data_dir data --output split_manifest.json
```

### 3. Data Augmentation
Performs offline augmentation on the training split. This creates a new directory `augmented_data/` and a new manifest.

```bash
python src/augment.py --manifest split_manifest.json --output_dir augmented_data
```

### 4. Training
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

### 5. Evaluation & Inference

**Model Evaluation:**
Calculate accuracy, precision/recall, and generate a confusion matrix on the test set.
```bash
python src/evaluate.py --config configs/custom_cnn.yaml --weights outputs/custom_cnn/best_model.keras
```
*Outputs are saved to the model's output directory (e.g., `outputs/custom_cnn/evaluation_results.json`).*

**Single Image Inference:**
Predict the class of a single image using a trained model.
```bash
python src/inference.py data/OBJ_001/001.jpg outputs/custom_cnn/best_model.keras
```
*Arguments:* `python src/inference.py <image_path> <model_path>`

## Available Models

Configuration files can be found in `configs/`:

| Config File | Model Type |
| :--- | :--- |
| `custom_cnn.yaml` | Custom CNN |
| `efficientnetv2s.yaml` | EfficientNetV2S |
| `convnexttiny.yaml` | ConvNeXtTiny |
| `mobilenetv2.yaml` | MobileNetV2 |
| `mobilenetv3large.yaml`| MobileNetV3Large |
| `mobilenetv3small.yaml`| MobileNetV3Small |

## Notes
- **Path Resolution:** Scripts in `src/` automatically resolve the project root, so they can be run from anywhere in the project tree.
- **Normalization:** 
  - `src/dataset.py` loads images as `float32` [0-255].
  - `src/models.py` handles normalization (e.g., `Rescaling(1./255)` or model-specific preprocessing).
