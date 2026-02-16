import argparse
import json
import os
import random
import sys
from pathlib import Path

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_arguments():
    parser = argparse.ArgumentParser(description="Scan dataset and create a train/val/test split manifest.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to root data folder containing class subfolders")
    parser.add_argument("--output", type=str, required=True, help="Path for output JSON manifest")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train/val/test ratios, must sum to 1.0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")
    return parser.parse_args()

def validate_ratios(ratios):
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"Error: Ratios {ratios} do not sum to 1.0")
        sys.exit(1)

def discover_images(data_dir):
    # Resolve data_dir relative to PROJECT_ROOT
    data_path = (PROJECT_ROOT / data_dir).resolve()
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' (resolved to '{data_path}') does not exist.")
        sys.exit(1)
        
    print(f"Project root determined as: {PROJECT_ROOT}")
    print(f"Scanning data directory: {data_path}")

    class_images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Sort for deterministic processing order before shuffling
    try:
        subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    except OSError as e:
        print(f"Error accessing data directory: {e}")
        sys.exit(1)

    for subdir in subdirs:
        class_name = subdir.name
        images = []
        try:
            files = sorted([f for f in subdir.iterdir() if f.is_file()])
            for file_path in files:
                if file_path.suffix.lower() in valid_extensions:
                    # Store path relative to project root
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    images.append(rel_path)
        except OSError as e:
            print(f"Warning: Error accessing subdirectory {subdir}: {e}")
            continue

        if images:
            class_images[class_name] = images
        else:
            print(f"Warning: No valid images found in class '{class_name}'. Skipping.")

    if not class_images:
        print("Error: No classes with images found.")
        sys.exit(1)

    return class_images

def create_splits(class_images, ratios, seed):
    random.seed(seed)
    train_ratio, val_ratio, test_ratio = ratios
    
    class_names = sorted(class_images.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    min_images = float('inf')
    max_images = 0

    for class_name in class_names:
        images = class_images[class_name]
        # Shuffle a copy to preserve original order in class_images
        shuffled_images = list(images)
        random.shuffle(shuffled_images)
        
        n_images = len(shuffled_images)
        min_images = min(min_images, n_images)
        max_images = max(max_images, n_images)
        
        train_end = int(n_images * train_ratio)
        val_end = train_end + int(n_images * val_ratio)
        
        train_imgs = shuffled_images[:train_end]
        val_imgs = shuffled_images[train_end:val_end]
        test_imgs = shuffled_images[val_end:]
        
        label_idx = class_to_idx[class_name]
        
        for img_path in train_imgs:
            splits["train"].append({"path": img_path, "label": class_name, "label_idx": label_idx})
        for img_path in val_imgs:
            splits["val"].append({"path": img_path, "label": class_name, "label_idx": label_idx})
        for img_path in test_imgs:
            splits["test"].append({"path": img_path, "label": class_name, "label_idx": label_idx})
            
    return splits, class_to_idx, min_images, max_images

def save_manifest(output_path, data_dir, ratios, seed, class_to_idx, splits):
    # Resolve paths
    abs_data_dir = (PROJECT_ROOT / data_dir).resolve()
    abs_output_path = (PROJECT_ROOT / output_path).resolve()

    metadata = {
        "data_dir": str(abs_data_dir),
        "num_classes": len(class_to_idx),
        "total_images": sum(len(s) for s in splits.values()),
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2]
        },
        "seed": seed,
        "split_counts": {k: len(v) for k, v in splits.items()}
    }
    
    manifest = {
        "metadata": metadata,
        "class_to_idx": class_to_idx,
        "splits": splits
    }
    
    os.makedirs(abs_output_path.parent, exist_ok=True)
    
    with open(abs_output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

def main():
    args = parse_arguments()
    validate_ratios(args.ratios)
    
    print(f"Scanning directory: {args.data_dir}")
    class_images = discover_images(args.data_dir)
    
    print(f"Creating splits with ratios: {args.ratios} and seed: {args.seed}")
    splits, class_to_idx, min_imgs, max_imgs = create_splits(class_images, args.ratios, args.seed)
    
    save_manifest(args.output, args.data_dir, args.ratios, args.seed, class_to_idx, splits)
    
    total_images = sum(len(s) for s in splits.values())
    print("\nSummary:")
    print(f"  Num classes: {len(class_to_idx)}")
    print(f"  Total images: {total_images}")
    print(f"  Split counts: {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, 'test': {len(splits['test'])}}}")
    print(f"  Images per class: min={min_imgs}, max={max_imgs}")
    print(f"  Manifest saved to: {args.output}")

if __name__ == "__main__":
    main()
