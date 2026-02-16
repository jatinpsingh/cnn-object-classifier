import argparse
import json
import os
import random
import sys
import shutil
from pathlib import Path
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform offline augmentation on training images.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to split manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save augmented images")
    parser.add_argument("--factor", type=int, default=3, help="Number of augmented copies per original image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def load_manifest(manifest_path):
    # manifest_path should already be resolved by main if necessary
    path = Path(manifest_path)
    if not path.exists():
        print(f"Error: Manifest file '{manifest_path}' does not exist.")
        sys.exit(1)
    
    with open(path, 'r') as f:
        return json.load(f)

def get_augmentation_pipeline():
    return A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.5),
        A.GaussNoise(p=0.3),
        A.HueSaturationValue(p=0.4),
    ])

def save_manifest(output_path, manifest_data):
    # Ensure parent dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)

def main():
    args = parse_arguments()
    
    if args.factor < 1:
        print("Error: Factor must be >= 1")
        sys.exit(1)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Resolve manifest path
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        # Try resolving relative to PROJECT_ROOT
        resolved_path = PROJECT_ROOT / args.manifest
        if resolved_path.exists():
            print(f"Manifest '{args.manifest}' not found in CWD, found at {resolved_path}")
            manifest_path = resolved_path
        else:
            print(f"Error: Manifest file '{args.manifest}' does not exist.")
            sys.exit(1)
            
    manifest = load_manifest(manifest_path)
    
    # We no longer rely on metadata.data_dir for project root.
    print(f"Project root determined as: {PROJECT_ROOT}")
    
    train_split = manifest["splits"]["train"]
    original_train_count = len(train_split)
    
    print(f"Loaded manifest with {original_train_count} training images.")
    print(f"Augmentation factor: {args.factor}")
    print(f"Output directory: {args.output_dir}")
    
    transform = get_augmentation_pipeline()
    
    # Resolve output directory physically relative to project root
    output_base_dir = (PROJECT_ROOT / args.output_dir).resolve()
    
    if output_base_dir.exists():
        print(f"Cleaning output directory: {output_base_dir}")
        shutil.rmtree(output_base_dir)
        
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    augmented_entries = []
    
    # Process training images
    print("Augmenting images...")
    for entry in tqdm(train_split):
        # Entry path is relative to project root (as per prepare_data.py)
        # We need absolute path to read it
        img_rel_path = entry["path"]
        img_abs_path = PROJECT_ROOT / img_rel_path
        
        label = entry["label"]
        label_idx = entry["label_idx"]
        
        # Read image
        image = cv2.imread(str(img_abs_path))
        if image is None:
            print(f"Warning: Could not read image {img_abs_path}. Skipping.")
            continue
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create class subdirectory in output dir
        class_dir = output_base_dir / label
        class_dir.mkdir(exist_ok=True)
        
        original_stem = Path(img_abs_path).stem
        
        for i in range(args.factor):
            try:
                # Apply augmentation
                augmented = transform(image=image)["image"]
                
                # Convert back to BGR for saving with OpenCV
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Check size and resize if needed
                if augmented_bgr.shape[0] != 224 or augmented_bgr.shape[1] != 224:
                    augmented_bgr = cv2.resize(augmented_bgr, (224, 224))
                
                out_filename = f"aug_{original_stem}_{i}.jpg"
                out_path_abs = class_dir / out_filename
                
                cv2.imwrite(str(out_path_abs), augmented_bgr)
                
                # Store path relative to project root
                out_path_rel = os.path.relpath(out_path_abs, PROJECT_ROOT)
                
                augmented_entries.append({
                    "path": str(out_path_rel),
                    "label": label,
                    "label_idx": label_idx
                })
            except Exception as e:
                print(f"Warning: Error augmenting {img_abs_path} (iter {i}): {e}")
                continue

    # Update manifest
    new_train_split = train_split + augmented_entries
    
    augmented_count = len(augmented_entries)
    total_train_count = len(new_train_split)
    
    manifest["splits"]["train"] = new_train_split
    
    # Update metadata
    manifest["metadata"]["total_images"] = (
        total_train_count + 
        len(manifest["splits"]["val"]) + 
        len(manifest["splits"]["test"])
    )
    manifest["metadata"]["split_counts"]["train"] = total_train_count
    
    manifest["metadata"]["augmentation"] = {
        "factor": args.factor,
        "original_train_count": original_train_count,
        "augmented_count": augmented_count,
        "total_train_count": total_train_count
    }
    
    # Construct output manifest path
    # Use resolved manifest_path to determine where to save the output manifest
    output_manifest_name = manifest_path.stem + "_augmented" + manifest_path.suffix
    output_manifest_path = manifest_path.parent / output_manifest_name
    
    save_manifest(output_manifest_path, manifest)
    
    print("\nSummary:")
    print(f"  Original train count: {original_train_count}")
    print(f"  Augmented count:      {augmented_count}")
    print(f"  New total train:      {total_train_count}")
    print(f"  Val count:            {len(manifest['splits']['val'])} (unchanged)")
    print(f"  Test count:           {len(manifest['splits']['test'])} (unchanged)")
    print(f"  Manifest saved to:    {output_manifest_path}")
    print(f"  Augmented images in:  {args.output_dir}")

if __name__ == "__main__":
    main()
