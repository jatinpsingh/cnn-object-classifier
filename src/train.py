import argparse
import json
import os
import sys
import yaml
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src to python path to allow imports if run from root
sys.path.append(str(PROJECT_ROOT / 'src'))

from models import build_model, load_config
from dataset import create_datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN model based on a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Optional overrides
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--early_stopping_patience", type=int, help="Override early stopping patience")
    
    return parser.parse_args()

def update_config(config, args):
    """Updates the config dictionary with CLI arguments."""
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.early_stopping_patience is not None:
        config['training']['early_stopping_patience'] = args.early_stopping_patience
    return config

def get_callbacks(config, output_dir):
    """Creates a list of Keras callbacks based on the config."""
    callbacks = []
    callback_names = config['training'].get('callbacks', [])
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)

    if "early_stopping" in callback_names:
        patience = config['training'].get('early_stopping_patience', 10)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ))
        
    if "model_checkpoint" in callback_names:
        # Save best model
        checkpoint_path = output_dir / "best_model.keras"
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ))
        
    if "reduce_lr" in callback_names:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ))
        
    return callbacks

def main():
    args = parse_args()
    
    # Load and update config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    config = load_config(str(config_path))
    config = update_config(config, args)
    
    # Resolve output directory
    output_dir_rel = config['output']['model_dir']
    output_dir = (PROJECT_ROOT / output_dir_rel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Output Directory: {output_dir}")
    
    # Load Data
    manifest_path_rel = config['data']['manifest']
    manifest_path = (PROJECT_ROOT / manifest_path_rel).resolve()
    
    batch_size = config['data']['batch_size']
    img_size = tuple(config['data']['img_size'])
    
    print("Loading datasets...")
    train_ds, val_ds, test_ds, num_classes = create_datasets(
        manifest_path=str(manifest_path),
        batch_size=batch_size,
        img_size=img_size,
        shuffle=True, # Always shuffle train
        prefetch=True
    )
    
    # Build Model
    print(f"Building model for {num_classes} classes...")
    model = build_model(config, num_classes)
    
    # Setup Callbacks
    callbacks = get_callbacks(config, output_dir)
    
    # Train
    epochs = config['training']['epochs']
    print(f"Starting training for {epochs} epochs...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save Results
    print("Saving results...")
    
    # 1. Save final model
    final_model_path = output_dir / "final_model.keras"
    model.save(str(final_model_path))
    
    # 2. Save history
    history_path = output_dir / "history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
        
    # 3. Save used config
    config_save_path = output_dir / "config_used.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # Print Summary
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*30)
    print("TRAINING COMPLETE")
    print("="*30)
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"Total Epochs Run: {len(history.history['loss'])}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Output Directory: {output_dir}")
    print("="*30)

if __name__ == "__main__":
    main()
