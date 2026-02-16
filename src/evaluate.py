import argparse
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add src to python path to allow importing dataset
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import create_datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--weights", required=True, help="Path to saved model weights (.keras file)")
    args = parser.parse_args()

    # 1. Load Config
    config_path = Path(args.config)
    if not config_path.exists():
        resolved_path = PROJECT_ROOT / args.config
        if resolved_path.exists():
            config_path = resolved_path
        else:
            print(f"Error: Config file not found at {args.config} or {resolved_path}")
            return

    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # 2. Load Data
    manifest_path = PROJECT_ROOT / config['data']['manifest']
    img_size = tuple(config['data']['img_size'])
    batch_size = config['data']['batch_size']
    
    print(f"Loading test data from manifest: {manifest_path}")
    
    # Load class names from manifest
    if not manifest_path.exists():
         raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    class_to_idx = manifest.get('class_to_idx', {})
    # Invert to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Ensure class names are sorted by index
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Load Test Dataset
    _, _, test_ds, num_classes = create_datasets(
        manifest_path=str(manifest_path),
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False, 
        prefetch=False # Disable prefetch for evaluation to simplify debugging if needed
    )
    
    if test_ds is None:
        print("Error: Test dataset is empty or could not be loaded.")
        return

    # 3. Load Model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        resolved_path = PROJECT_ROOT / args.weights
        if resolved_path.exists():
            weights_path = resolved_path
        else:
            print(f"Error: Weights file not found at {args.weights} or {resolved_path}")
            return
        
    print(f"Loading model from {weights_path}...")
    try:
        model = tf.keras.models.load_model(weights_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 4. Run Inference
    print("Running inference on test set...")
    y_true = []
    y_pred_probs = []
    
    # Iterate over the dataset
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        # labels are one-hot encoded in dataset.py
        y_true.append(np.argmax(labels.numpy(), axis=1))
        
    if not y_true:
        print("No test data found.")
        return

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5. Compute Metrics
    print("Computing metrics...")
    
    # Test Accuracy
    test_accuracy = np.mean(y_true == y_pred)
    
    # Top-5 Accuracy
    # Check if true label index is in the top 5 predicted indices
    # argsort sorts in ascending order, so take last 5
    top_5_indices = np.argsort(y_pred_probs, axis=1)[:, -5:]
    correct_top_5 = [true_label in top_5_indices[i] for i, true_label in enumerate(y_true)]
    top_5_acc = np.mean(correct_top_5)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top_5_acc:.4f}")
    
    # Classification Report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification Report:")
    print(report_text)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 6. Save Outputs
    output_model_dir = config['output']['model_dir']
    output_dir = PROJECT_ROOT / output_model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results = {
        "test_accuracy": float(test_accuracy),
        "top_5_accuracy": float(top_5_acc),
        "num_test_samples": len(y_true),
        "per_class_report": report_dict
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Save Confusion Matrix Plot
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {config.get('experiment_name', 'Experiment')}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    print("-" * 30)
    print(f"Evaluation complete.")
    print(f"Results saved to: {results_path}")
    print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()
