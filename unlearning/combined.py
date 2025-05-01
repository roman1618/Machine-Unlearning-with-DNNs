import tensorflow as tf
from tensorflow import keras
import numpy as np

from .layer_reset import reinitialize_model_layers
from training.trainer import train_model # Assuming train_model handles fine-tuning

# Optional: Add noise function if needed, similar to vision_confuser
def add_gaussian_noise_to_layers(model, std_dev, layer_names=None):
    """Adds Gaussian noise to the weights of specified layers."""
    if std_dev <= 0:
        return # No noise to add

    print(f"Adding Gaussian noise with std_dev={std_dev}")
    for layer in model.layers:
        # Check if layer is specified or if all layers should be modified
        if layer_names is None or layer.name in layer_names:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                original_weights = layer.get_weights()
                noisy_weights = []
                for w in original_weights:
                    noise = tf.random.normal(shape=tf.shape(w), mean=0.0, stddev=std_dev, dtype=w.dtype)
                    noisy_weights.append(w + noise)
                try:
                    layer.set_weights(noisy_weights)
                    print(f"  - Added noise to layer: {layer.name}")
                except Exception as e:
                    print(f"  - Could not set weights for layer {layer.name}: {e}")
            # Handle nested models/layers
            elif hasattr(layer, 'layers'):
                 add_gaussian_noise_to_layers(layer, std_dev, layer_names)


def combined_unlearn(
    model,
    retain_ds,
    val_ds=None,
    layers_to_reset=None,
    noise_std_dev=0.0,
    noise_layers=None, # Specify layer names for noise, None for all
    fine_tune_epochs=0,
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    callbacks=None,
    strategy=None # Pass distribution strategy if used
):
    """
    Performs unlearning by optionally resetting layers, adding noise, and fine-tuning.

    Args:
        model: The Keras model to unlearn from.
        retain_ds: The dataset containing only the data to be retained.
        val_ds: Optional validation dataset for fine-tuning.
        layers_to_reset: List of layer names or indices to reinitialize.
        noise_std_dev: Standard deviation for Gaussian noise to add to weights.
        noise_layers: List of layer names to add noise to (if None, adds to all possible).
        fine_tune_epochs: Number of epochs to fine-tune on the retain dataset.
        optimizer: Optimizer for fine-tuning.
        loss: Loss function for fine-tuning.
        metrics: Metrics for fine-tuning.
        callbacks: Callbacks for fine-tuning.
        strategy: TensorFlow distribution strategy.

    Returns:
        The unlearned Keras model.
    """
    unlearn_model = tf.keras.models.clone_model(model)
    unlearn_model.set_weights(model.get_weights())

    # 1. Reinitialize specified layers
    if layers_to_reset:
        print("--- Reinitializing Layers ---")
        reinitialize_model_layers(unlearn_model, layer_names=layers_to_reset)

    # 2. Add noise to specified layers
    if noise_std_dev > 0:
        print("--- Adding Noise --- ")
        add_gaussian_noise_to_layers(unlearn_model, noise_std_dev, layer_names=noise_layers)

    # 3. Fine-tune on the retain dataset
    if fine_tune_epochs > 0:
        print(f"--- Fine-tuning for {fine_tune_epochs} epochs ---")
        # Re-compile the model within the strategy scope if applicable
        compile_args = {
            'optimizer': optimizer,
            'loss': loss,
            'metrics': metrics
        }
        if strategy:
            with strategy.scope():
                unlearn_model.compile(**compile_args)
        else:
            unlearn_model.compile(**compile_args)

        # Use the train_model function for fine-tuning
        history = train_model(
            unlearn_model,
            train_ds=retain_ds,
            val_ds=val_ds,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            strategy=strategy # Pass strategy here as well
        )
        print("Fine-tuning complete.")
    else:
        print("Skipping fine-tuning.")

    return unlearn_model
