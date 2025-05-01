import tensorflow as tf
from tensorflow.keras import metrics

def change_model_names(model, name, suffix):
    """Recursively changes all layer names in a Keras model by adding a suffix.

    Args:
        model (tf.keras.Model): The Keras model.
        name (str): The base name for the model itself.
        suffix (str): The string suffix to add to the layer names.
    """
    model._name = name # Set the model's top-level name
    # Iterate through the layers of the model
    for layer in model.layers:
        # Append suffix, avoiding duplicate suffixes if function is called multiple times
        if not layer.name.endswith(suffix):
            layer._name += suffix
        # If the layer is a backbone or contains sublayers, recursively change names
        if hasattr(layer, 'layers') and isinstance(layer.layers, list):
             # Check if it's a Keras model/layer with sublayers
            change_model_names(layer, layer.name, suffix) # Pass current layer name as base
    print(f"Updated layer names for model '{name}' with suffix '{suffix}'.")

def clone_model(original_model, strategy):
    """Clones a Keras model and compiles it within a distribution strategy.

    Args:
        original_model (tf.keras.Model): The model to clone.
        strategy (tf.distribute.Strategy): The distribution strategy.

    Returns:
        tf.keras.Model: The cloned and compiled model.
    """
    with strategy.scope():
        cloned_model = tf.keras.models.clone_model(original_model)
        cloned_model.set_weights(original_model.get_weights()) # Copy weights
        # Re-compile the cloned model explicitly defining the metrics
        cloned_model.compile(
            optimizer=tf.keras.optimizers.Adam(), # Create a new optimizer instance
            loss=original_model.loss, # Use the same loss function
            # Explicitly define the accuracy metric
            metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
            jit_compile=True
        )
    print(f"Cloned and re-compiled model '{original_model.name}' with explicit accuracy metric.")
    return cloned_model
