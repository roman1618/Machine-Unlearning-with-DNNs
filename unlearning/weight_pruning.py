import tensorflow as tf
import numpy as np

def prune_weights_by_magnitude(model, pruning_percentage):
    """
    Prunes weights in a Keras model based on magnitude.

    Sets weights with the smallest absolute magnitude to zero.

    Args:
        model: The Keras model to prune.
        pruning_percentage: The percentage of weights to prune (0.0 to 1.0).
    """
    if not 0.0 <= pruning_percentage <= 1.0:
        raise ValueError("Pruning percentage must be between 0.0 and 1.0")

    if pruning_percentage == 0.0:
        print("Pruning percentage is 0. No weights will be pruned.")
        return model # Or return a copy if immutability is desired

    all_weights = []
    layer_weight_shapes = []
    layer_indices = [] # Keep track of which layers had weights

    # Collect all trainable weights and their shapes
    for i, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer_weights: # Only consider layers with weights
            layer_indices.append(i)
            current_layer_shapes = []
            for w in layer_weights:
                if w.ndim > 1: # Typically kernels/embeddings, ignore biases/batchnorm params for magnitude pruning
                    all_weights.append(w.flatten())
                    current_layer_shapes.append(w.shape)
                else:
                     # Keep non-prunable weights (like biases) as is, store None shape
                    current_layer_shapes.append(None)
            layer_weight_shapes.append(current_layer_shapes)


    if not all_weights:
        print("No prunable weights found in the model.")
        return model

    # Concatenate all weights into a single array for percentile calculation
    flat_weights = np.concatenate(all_weights)
    abs_weights = np.abs(flat_weights)

    # Determine the magnitude threshold
    threshold = np.percentile(abs_weights, pruning_percentage * 100)
    print(f"Pruning threshold (magnitude): {threshold}")

    # Apply pruning
    pruned_weights_flat = np.where(abs_weights >= threshold, flat_weights, 0.0)

    # Reshape pruned weights and update the model
    start_idx = 0
    weight_layer_idx = 0
    for i, layer in enumerate(model.layers):
         if i in layer_indices:
            original_layer_weights = layer.get_weights()
            new_layer_weights = []
            shapes_for_layer = layer_weight_shapes[weight_layer_idx]
            prunable_weight_idx = 0 # Index within the prunable weights of this layer

            for j, w_orig in enumerate(original_layer_weights):
                shape = shapes_for_layer[j]
                if shape is not None: # This was a prunable weight
                    num_elements = np.prod(shape)
                    end_idx = start_idx + num_elements
                    # Get the pruned weights for this tensor
                    pruned_tensor_flat = pruned_weights_flat[start_idx:end_idx]
                    pruned_tensor = pruned_tensor_flat.reshape(shape)
                    new_layer_weights.append(pruned_tensor.astype(w_orig.dtype)) # Ensure dtype matches
                    start_idx = end_idx
                    prunable_weight_idx += 1
                else:
                    # This weight was not pruned (e.g., bias), keep original
                    new_layer_weights.append(w_orig)

            layer.set_weights(new_layer_weights)
            weight_layer_idx += 1

    print(f"Pruned {pruning_percentage*100:.2f}% of weights by magnitude.")
    return model # Return the modified model
