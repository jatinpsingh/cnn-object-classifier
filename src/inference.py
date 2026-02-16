import argparse
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.keras file)")
    parser.add_argument("--manifest", type=str, default="split_manifest.json", help="Path to the manifest file for class mappings")
    parser.add_argument("--img_size", type=int, default=224, help="Target image size (e.g., 224 for 224x224)")
    return parser.parse_args()

def load_class_names(manifest_path):
    """Loads class names from the manifest JSON."""
    path = Path(manifest_path)
    if not path.exists():
        # Try finding it relative to PROJECT_ROOT
        path = PROJECT_ROOT / manifest_path
        if not path.exists():
            print(f"Error: Manifest file not found at {manifest_path} or {path}")
            sys.exit(1)
            
    with open(path, 'r') as f:
        manifest = json.load(f)
        
    class_to_idx = manifest.get('class_to_idx', {})
    if not class_to_idx:
        print("Error: 'class_to_idx' mapping not found in manifest.")
        sys.exit(1)
        
    # Invert to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

def preprocess_image(image_path, img_size=(224, 224)):
    """Reads and preprocesses an image for inference."""
    path = Path(image_path)
    if not path.exists():
         # Try relative to PROJECT_ROOT
        path = PROJECT_ROOT / image_path
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

    image_content = tf.io.read_file(str(path))
    # expand_animations=False fixes issues with some image formats in newer TF versions
    image = tf.image.decode_image(image_content, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    # Cast to float32 (0-255 range, model handles normalization usually)
    image = tf.cast(image, tf.float32)
    # Add batch dimension
    image = tf.expand_dims(image, 0)
    return image

def main():
    args = parse_args()
    
    # 1. Load Class Mappings
    idx_to_class = load_class_names(args.manifest)
    
    # 2. Load Model
    model_path = Path(args.model_path)
    if not model_path.exists():
        model_path = PROJECT_ROOT / args.model_path
        if not model_path.exists():
            print(f"Error: Model file not found at {args.model_path}")
            sys.exit(1)
            
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Preprocess Image
    try:
        img_tensor = preprocess_image(args.image_path, (args.img_size, args.img_size))
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
        
    # 4. Run Inference
    print("Running inference...")
    predictions = model.predict(img_tensor, verbose=0)
    
    # 5. Process Results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_label = idx_to_class.get(predicted_idx, "Unknown")
    
    print("-" * 30)
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence:      {confidence:.4f} ({confidence*100:.2f}%)")
    print("-" * 30)

if __name__ == "__main__":
    main()
