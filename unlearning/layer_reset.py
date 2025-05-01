import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Conv2D

def reinitialize_layer_weights(layer):
    """Reinitializes the weights of a single Keras layer."""
    if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
        # Get initializers
        kernel_initializer = layer.kernel_initializer
        bias_initializer = layer.bias_initializer

        # Get shapes
        kernel_shape = layer.kernel.shape
        bias_shape = layer.bias.shape

        # Generate new weights and biases
        new_kernel = kernel_initializer(shape=kernel_shape)
        new_bias = bias_initializer(shape=bias_shape)

        # Set new weights
        layer.set_weights([new_kernel, new_bias])
        print(f"Reinitialized weights for layer: {layer.name}")

def reinitialize_model_layers(model, layer_indices=None, layer_names=None):
    """
    Reinitializes weights for specified layers in a Keras model.

    Args:
        model: The Keras model.
        layer_indices: A list of indices of the layers to reinitialize.
        layer_names: A list of names of the layers to reinitialize.
                     Only one of layer_indices or layer_names should be provided.
    """
    if layer_indices is not None and layer_names is not None:
        raise ValueError("Provide either layer_indices or layer_names, not both.")

    if layer_indices is not None:
        for i in layer_indices:
            if 0 <= i < len(model.layers):
                reinitialize_layer_weights(model.layers[i])
            else:
                print(f"Warning: Layer index {i} is out of bounds.")
    elif layer_names is not None:
        reinitialized_count = 0
        for layer in model.layers:
            if layer.name in layer_names:
                reinitialize_layer_weights(layer)
                reinitialized_count += 1
        if reinitialized_count != len(layer_names):
             print(f"Warning: Found and reinitialized {reinitialized_count} layers, but {len(layer_names)} names were provided.")
    else:
        print("No layers specified for reinitialization.")

def reinitialize_last_layer(model):
    """Reinitializes the weights of the last layer of a Keras model."""
    if model.layers:
        reinitialize_layer_weights(model.layers[-1])
    else:
        print("Model has no layers to reinitialize.")

def vision_confuser(model, std=0.1):
    """Adds Gaussian noise to the weights of Conv2D layers and the final Dense layer.

    Args:
        model (tf.keras.Model): The Keras model to modify.
        std (float): The standard deviation of the Gaussian noise to add.
    """
    print(f"Applying vision confusion with std={std} to model '{model.name}'...")
    # Assuming the ResNet18 backbone is the first layer if it exists
    try:
        # Attempt to get backbone by common name patterns
        if model.layers[0].name.startswith('resnet') or 'backbone' in model.layers[0].name:
            backbone = model.layers[0]
            print(f"  Found backbone layer: {backbone.name}")
            layers_to_confuse = backbone.layers
        else:
             # If no clear backbone, apply to all layers in the main model sequence
             print("  No standard backbone layer found, applying to top-level Conv2D layers.")
             layers_to_confuse = model.layers
    except IndexError:
        print("  Model has no layers.")
        layers_to_confuse = []


    confused_conv_count = 0
    for layer in layers_to_confuse:
        # Confuse Conv2D layers within the backbone or model
        if isinstance(layer, Conv2D):
            weights = layer.get_weights()
            new_weights = []
            # Iterate over each weight array (kernel, bias) in the layer
            for i, arr in enumerate(weights):
                # Generate random numbers from a normal distribution
                random_numbers = np.random.normal(0, std, size=arr.shape).astype(arr.dtype) # Match dtype
                # Add the random numbers to the original array
                new_weights.append(arr + random_numbers)
            if new_weights:
                layer.set_weights(new_weights)
                confused_conv_count += 1

    print(f"  Confused {confused_conv_count} Conv2D layers.")

    # Always confuse the last layer if it's a Dense layer (typical classifier head)
    if model.layers and isinstance(model.layers[-1], keras.layers.Dense):
        print(f"  Confusing final Dense layer: {model.layers[-1].name}")
        weights = model.layers[-1].get_weights()
        new_weights = []
        for i, arr in enumerate(weights):
            random_numbers = np.random.normal(0, std, size=arr.shape).astype(arr.dtype)
            new_weights.append(arr + random_numbers)
        if new_weights:
            model.layers[-1].set_weights(new_weights)
            print("  Final Dense layer confused.")
    else:
        print("  Last layer is not a Dense layer or model has no layers, skipping final layer confusion.")
