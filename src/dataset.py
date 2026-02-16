import tensorflow as tf
import json
from pathlib import Path

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_datasets(
    manifest_path="split_manifest_augmented.json",
    batch_size=32,
    img_size=(224, 224),
    shuffle=True,
    prefetch=True,
    shuffle_buffer=10000
):
    """
    Creates TensorFlow datasets for train, val, and test splits from a JSON manifest.

    Args:
        manifest_path (str): Path to the manifest JSON file.
        batch_size (int): Batch size.
        img_size (tuple): Target image size (height, width).
        shuffle (bool): Whether to shuffle the train split.
        prefetch (bool): Whether to prefetch batches.
        shuffle_buffer (int): Buffer size for shuffling.

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Test dataset.
        num_classes (int): Number of classes.
    """
    
    # Step 1: Load Manifest
    # manifest_path is assumed to be either absolute or relative to CWD.
    manifest_path_obj = Path(manifest_path)
    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    class_to_idx = manifest.get('class_to_idx', {})
    num_classes = len(class_to_idx)
    
    # We ignore metadata.data_dir for root derivation.
    # Paths in manifest are relative to PROJECT_ROOT.
    
    splits = manifest.get('splits', {})
    
    # Helper to prepare lists of absolute paths and label indices
    def get_paths_and_labels(split_name):
        if split_name not in splits:
            return [], []
        
        items = splits[split_name]
        paths = []
        labels = []
        for item in items:
            # item['path'] is relative to PROJECT_ROOT, e.g., "data/OBJ_001/img.jpg"
            abs_path = PROJECT_ROOT / item['path']
            paths.append(str(abs_path))
            labels.append(item['label_idx'])
        return paths, labels

    # Step 2: Build Parse Function
    @tf.function
    def parse_function(file_path, label):
        # Load image
        image_content = tf.io.read_file(file_path)
        # channels=3 ensures 3 channels (RGB)
        image = tf.image.decode_image(image_content, channels=3, expand_animations=False) 
        image = tf.image.resize(image, img_size)
        # Cast to float32 (0-255 range)
        image = tf.cast(image, tf.float32)
        
        # One-hot encode label
        label = tf.one_hot(label, depth=num_classes)
        
        return image, label

    # Step 3: Build Dataset for Each Split
    datasets = {}
    split_names = ['train', 'val', 'test']
    
    print(f"Loaded manifest from {manifest_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Project root determined as: {PROJECT_ROOT}")

    for split in split_names:
        paths, labels = get_paths_and_labels(split)
        count = len(paths)
        print(f"Split '{split}': {count} images")

        if count == 0:
            # Handle empty split gracefully
            ds = tf.data.Dataset.from_tensor_slices((
                tf.constant([], dtype=tf.string),
                tf.constant([], dtype=tf.int32)
            ))
        else:
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            
        # Apply mapping
        ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle (train only)
        if split == 'train' and shuffle and count > 0:
            ds = ds.shuffle(buffer_size=shuffle_buffer)
            
        # Batch
        ds = ds.batch(batch_size)
        
        # Prefetch
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
            
        datasets[split] = ds

    return datasets['train'], datasets['val'], datasets['test'], num_classes
