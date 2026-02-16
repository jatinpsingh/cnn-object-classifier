import tensorflow as tf
import yaml
from pathlib import Path

def load_config(yaml_path):
    """
    Loads a YAML configuration file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        config (dict): Configuration dictionary.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _build_custom_cnn(config, num_classes):
    """
    Builds a custom CNN model based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        num_classes (int): Number of classes.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model_config = config['model']
    training_config = config['training']

    input_shape = tuple(model_config.get('input_shape', (224, 224, 3)))
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # Rescaling (Normalization)
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    # Convolutional Blocks
    filters_list = model_config.get('layers', [32, 64, 128, 256])
    
    for filters in filters_list:
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense Head
    head_units = model_config.get('head_units', 256)
    dropout_rate = model_config.get('dropout', 0.5)
    
    x = tf.keras.layers.Dense(head_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='custom_cnn')
    
    # Compilation
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
    loss = training_config.get('loss', 'categorical_crossentropy')
    metrics = training_config.get('metrics', ['accuracy'])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def _build_transfer_model(config, num_classes):
    """
    Builds a transfer learning model using MobileNet family.
    
    Args:
        config (dict): Configuration dictionary.
        num_classes (int): Number of classes.
        
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model_config = config['model']
    head_config = config.get('head', {})
    training_config = config['training']
    
    model_type = model_config['type']
    input_shape = tuple(model_config.get('input_shape', (224, 224, 3)))
    weights = model_config.get('weights', 'imagenet')
    freeze_base = model_config.get('freeze_base', True)
    
    # Map model types to their module for preprocessing
    model_map = {
        'MobileNetV2': (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
        'MobileNetV3Small': (tf.keras.applications.MobileNetV3Small, tf.keras.applications.mobilenet_v3.preprocess_input),
        'MobileNetV3Large': (tf.keras.applications.MobileNetV3Large, tf.keras.applications.mobilenet_v3.preprocess_input),
        'EfficientNetV2S': (tf.keras.applications.EfficientNetV2S, tf.keras.applications.efficientnet_v2.preprocess_input),
        'ConvNeXtTiny': (tf.keras.applications.ConvNeXtTiny, tf.keras.applications.convnext.preprocess_input),
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported transfer model type: {model_type}")
        
    BaseModelClass, preprocess_input = model_map[model_type]
    
    # Load base model
    base_model = BaseModelClass(weights=weights, include_top=False, input_shape=input_shape)
    base_model.trainable = not freeze_base
    
    # Build model using Functional API
    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False) # training=False ensures BatchNormalization layers run in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    head_units = head_config.get('units', 256)
    dropout_rate = head_config.get('dropout', 0.5)
    
    x = tf.keras.layers.Dense(head_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_type)
    
    # Compilation
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
    loss = training_config.get('loss', 'categorical_crossentropy')
    metrics = training_config.get('metrics', ['accuracy'])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print(f"Base Model: {model_type}")
    print(f"Base Model Trainable: {base_model.trainable}")
    
    return model

def build_model(config, num_classes):
    """
    Builds and compiles a Keras model based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        num_classes (int): Number of classes.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model_type = config['model']['type']
    
    if model_type == 'custom_cnn':
        model = _build_custom_cnn(config, num_classes)
    else:
        model = _build_transfer_model(config, num_classes)
        
    model.summary()
    
    return model
