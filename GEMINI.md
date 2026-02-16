# Agent Profile
- **Name:** worker
- **Role:** Coding agent for this project

# Environment
Always use the anaconda environment named "image_classification" for this project. Activate it before running any commands. Always prefer libraries which utilise dedicated gpu onboard. If you want to install any missing libraries, always use "image_classification" env, use "conda install first" and if that does not work, then use "pip install".

**Key Dependencies:**
- `tensorflow` (GPU version preferred)
- `albumentations` (for data augmentation)
- `opencv` (for image processing in augmentation)
- `pyyaml` (for configuration parsing)
- `tqdm` (for progress bars)
- `numpy`

# Context
Building a CNN-based object classification system.

# Dataset
- **Location:** `~/projects/deeplearning/image_classification/data/`
- **Structure:** One subfolder per class, 39 classes total
- **Class names:** 
  OBJ_001, OBJ_002, OBJ_003, OBJ_004, OBJ_005, OBJ_006, OBJ_007, OBJ_008, OBJ_009, OBJ_010, 
  OBJ_012, OBJ_016, OBJ_018, OBJ_019, OBJ_021, OBJ_022, OBJ_027, OBJ_028, OBJ_029, OBJ_031, 
  OBJ_061, OBJ_069, OBJ_090, OBJ_095, OBJ_107, OBJ_108, OBJ_111, OBJ_159, OBJ_208, OBJ_222, 
  OBJ_229, OBJ_230, OBJ_300, OBJ_311, OBJ_405, OBJ_786, OBJ_787, OBJ_788, OBJ_789
- **Details:** Each class has 100+ images, 224x224 PNG/JPG

# Project Structure
```
/home/jatin/projects/deeplearning/image_classification/
├── configs/
│   └── custom_cnn.yaml      # Configuration for custom CNN training
├── src/
│   ├── augment.py           # Offline data augmentation
│   ├── dataset.py           # TensorFlow dataset pipeline
│   ├── models.py            # Model factory and definition
│   ├── prepare_data.py      # Data discovery and split manifest creation
│   └── train.py             # Training script
├── data/                    # Original dataset
├── augmented_data/          # Generated augmented images
├── outputs/                 # Training artifacts (models, logs)
├── split_manifest.json      # Original data splits
└── split_manifest_augmented.json # Data splits including augmented data
```

# Workflows

## 1. Data Preparation
- **Script:** `src/prepare_data.py`
- **Purpose:** Scans the `data/` directory and creates a JSON manifest defining train/val/test splits.
- **Usage:**
  ```bash
  python src/prepare_data.py --data_dir data --output split_manifest.json
  ```

## 2. Data Augmentation
- **Script:** `src/augment.py`
- **Purpose:** Performs offline augmentation on the training split defined in a manifest.
- **Key Features:**
  - **Auto-Cleaning:** Automatically deletes and recreates the `output_dir` before generating new images to ensure a clean state.
  - **Path Resolution:** Attempts to resolve the manifest path relative to the current working directory first, then falls back to resolving it relative to `PROJECT_ROOT`.
- **Usage:**
  ```bash
  python src/augment.py --manifest split_manifest.json --output_dir augmented_data
  ```
  Produces `split_manifest_augmented.json`.

## 3. Training
- **Script:** `src/train.py`
- **Purpose:** Trains a model based on a YAML configuration.
- **Usage:**
  ```bash
  python src/train.py --config configs/custom_cnn.yaml
  ```
- **Overrides:** CLI arguments can override config values:
  ```bash
  python src/train.py --config configs/custom_cnn.yaml --epochs 100 --learning_rate 0.0005
  ```
- **Artifacts:** Saved to the configured `output.model_dir`:
  - `best_model.keras`: Model checkpoint with the lowest validation loss.
  - `final_model.keras`: Model state after the final epoch.
  - `history.json`: Training metrics (loss, accuracy) for every epoch.
  - `config_used.yaml`: The effective configuration used for the run (including CLI overrides).

# Conventions

## Path Resolution
- All scripts in `src/` derive the `PROJECT_ROOT` relative to their own location:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  ```
- **Robustness:** This derivation allows scripts to be executed from any working directory; they will always correctly locate the project root.
- All input/output paths (manifests, data dirs, output dirs) are resolved relative to `PROJECT_ROOT`.

## Data Pipeline & Normalization
- **`src/dataset.py`**: 
  - Loads images.
  - Resizes to target size (default 224x224).
  - Casts to `float32`.
  - **Does NOT normalize** pixel values (range remains 0-255).
- **`src/models.py`**:
  - Responsible for normalization.
  - The `custom_cnn` model includes a `Rescaling(1./255)` layer as its first layer.

## Configuration
- Training is config-driven using YAML files (e.g., `configs/custom_cnn.yaml`).
- Configs define:
  - Experiment name
  - Model architecture parameters
  - Data settings (manifest path, batch size)
  - Training hyperparameters (optimizer, LR, epochs, callbacks)
  - Output directory
