import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

def compute_losses(net, loader):
    """Computes per-sample losses for a model on a given dataset loader.

    Args:
        net (tf.keras.Model): The trained Keras model.
        loader (tf.data.Dataset): A tf.data.Dataset yielding batches of (inputs, targets).
                                  Should be batched.

    Returns:
        np.ndarray: An array containing the loss for each sample in the loader.
    """
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE # Compute loss per sample
    )
    all_losses = []
    print(f"Computing losses for model '{net.name}'...")
    # Determine the total number of batches if possible
    total_batches = tf.data.experimental.cardinality(loader).numpy()
    if total_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        total_batches = None # Cannot determine size beforehand

    for inputs, targets in tqdm(loader, total=total_batches, desc="Batch"):
        logits = net(inputs, training=False) # Inference mode
        losses = loss_fn(targets, logits).numpy() # Get losses as numpy array
        all_losses.extend(losses) # Use extend for list of losses

    print("Loss computation complete.")
    return np.array(all_losses)

def evaluate_model(model, datasets, dataset_names):
    """Evaluates a model on multiple datasets and returns accuracies.

    Args:
        model (tf.keras.Model): The model to evaluate.
        datasets (list): A list of tf.data.Dataset objects to evaluate on.
        dataset_names (list): A list of names corresponding to the datasets.

    Returns:
        dict: A dictionary mapping dataset names to their accuracy scores.
    """
    results = {}
    print(f"Evaluating model '{model.name}'...")
    if len(datasets) != len(dataset_names):
        raise ValueError("Number of datasets and names must match.")

    for name, ds in zip(dataset_names, datasets):
        print(f"  Evaluating on {name} set...")
        # Ensure the dataset is batched
        if ds.element_spec[0].shape.ndims == 3: # Check if images are unbatched (e.g., (32, 32, 3))
             ds_batched = ds.batch(128).prefetch(tf.data.AUTOTUNE) # Use a default batch size
             print(f"    Batched '{name}' dataset for evaluation.")
        else:
             ds_batched = ds

        loss, accuracy = model.evaluate(ds_batched, verbose=0)
        print(f"  {name} set accuracy: {accuracy * 100.0:.2f}%")
        results[name] = accuracy
    return results

# Add other metrics like cosine similarity if needed
