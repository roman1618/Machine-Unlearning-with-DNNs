import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm

from models.resnet18 import build_resnet18 # Assuming model build function is here
from training.trainer import train_model # Assuming training function is here
from data.cifar10 import normalize # Assuming normalize function is here

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128 # TODO: Consider making this configurable

def train_shard_models(
    sliced_datasets,
    val_ds,
    num_shards,
    epochs,
    model_base_path="shard_models",
    strategy=None
):
    """
    Trains one model for each shard of the dataset.

    Args:
        sliced_datasets: A list of tf.data.Dataset objects, one for each shard.
        val_ds: Validation dataset (tf.data.Dataset).
        num_shards: The number of shards (S).
        epochs: Number of epochs to train each shard model.
        model_base_path: Directory path to save the trained shard models.
        strategy: TensorFlow distribution strategy (optional).

    Returns:
        A list of trained Keras models, one for each shard.
    """
    shard_models = []
    os.makedirs(model_base_path, exist_ok=True)

    print(f"Training {num_shards} shard models...")
    for s in tqdm(range(num_shards), desc="Training Shards"):
        print(f"\n--- Training Shard {s+1}/{num_shards} ---")
        shard_ds = sliced_datasets[s]
        # Prepare dataset: normalize, shuffle, batch, prefetch
        shard_ds = shard_ds.map(normalize, num_parallel_calls=AUTOTUNE)
        shard_ds = shard_ds.shuffle(buffer_size=8 * BATCH_SIZE)
        shard_ds = shard_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

        # Build a new model for each shard
        if strategy:
            with strategy.scope():
                model = build_resnet18()
        else:
            model = build_resnet18()

        # Train the model
        train_model(model, shard_ds, val_ds=val_ds, epochs=epochs)

        # Save the model
        model_path = os.path.join(model_base_path, f"shard_{s}.keras")
        model.save(model_path, save_format="keras_v3")
        print(f"Saved model for shard {s} to {model_path}")
        shard_models.append(model)

    return shard_models


def sisa_unlearn(
    forget_indices,
    num_shards,
    original_datasets, # List of original shard datasets (before batching/prefetching)
    val_ds,
    epochs,
    model_base_path="shard_models",
    strategy=None
):
    """
    Performs SISA unlearning by retraining models for affected shards.

    Args:
        forget_indices: A list or set of global indices of samples to forget.
        num_shards: The total number of shards.
        original_datasets: List of original tf.data.Dataset shards (containing original indices).
                           Each element should be a tuple (dataset, original_indices_array).
        val_ds: Validation dataset.
        epochs: Number of epochs to retrain affected models.
        model_base_path: Directory path where shard models are saved/loaded.
        strategy: TensorFlow distribution strategy (optional).

    Returns:
        A list of all shard models (retrained or loaded).
    """
    shards_to_retrain = set()
    forget_indices_set = set(forget_indices)

    print("Identifying shards containing data to forget...")
    # This assumes original_datasets contains tuples (shard_data, shard_indices)
    # and shard_indices maps the shard's internal index to the global index.
    # This mapping needs to be created during the initial sharding process.
    # For now, we'll simulate finding the shards. A more robust implementation
    # would store this mapping explicitly.

    # Placeholder: Determine which shards contain the forget_indices
    # In a real scenario, you'd iterate through the metadata linking global indices to shards.
    # Example simulation (replace with actual logic):
    samples_per_shard = 50000 // num_shards # Approximate
    for idx in forget_indices_set:
        shard_index = idx // samples_per_shard
        if shard_index < num_shards:
            shards_to_retrain.add(shard_index)
        else: # Handle potential edge cases if division isn't exact
             shards_to_retrain.add(num_shards - 1)


    print(f"Shards identified for retraining: {list(shards_to_retrain)}")

    all_models = []
    for s in range(num_shards):
        model_path = os.path.join(model_base_path, f"shard_{s}.keras")
        if s in shards_to_retrain:
            print(f"\n--- Retraining Shard {s+1}/{num_shards} ---")
            # Need the actual data for the shard, excluding the forgotten samples.
            # This requires filtering the original_datasets[s]
            # Placeholder: Assume we have the correct filtered dataset `retrain_ds`
            # In reality: filter original_datasets[s] based on forget_indices_set

            # --- Filtering Logic Start (Conceptual) ---
            shard_data, shard_indices = original_datasets[s] # Assuming this structure
            retain_mask = np.isin(shard_indices, list(forget_indices_set), invert=True)

            # Filter the dataset - this depends on how shard_data is structured
            # If shard_data is (x, y) numpy arrays:
            # x_retain = shard_data[0][retain_mask]
            # y_retain = shard_data[1][retain_mask]
            # retrain_ds_unbatched = tf.data.Dataset.from_tensor_slices((x_retain, y_retain))

            # If shard_data is already a tf.data.Dataset, filtering is more complex:
            # We might need to recreate it or use tf.data.Dataset.filter
            # For simplicity, let's assume we can get the filtered data
            # This part needs careful implementation based on data structure!
            print(f"Warning: Dataset filtering for retraining shard {s} is conceptual.")
            # Using the original shard data for now as a placeholder
            retrain_ds_unbatched = shard_data
            # --- Filtering Logic End ---


            # Prepare dataset
            retrain_ds = retrain_ds_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
            retrain_ds = retrain_ds.shuffle(buffer_size=8 * BATCH_SIZE)
            retrain_ds = retrain_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

            # Build and train
            if strategy:
                with strategy.scope():
                    model = build_resnet18()
            else:
                model = build_resnet18()

            train_model(model, retrain_ds, val_ds=val_ds, epochs=epochs)
            model.save(model_path, save_format="keras_v3") # Overwrite old model
            print(f"Retrained and saved model for shard {s} to {model_path}")
            all_models.append(model)
        else:
            # Load existing model
            print(f"Loading existing model for shard {s} from {model_path}")
            if strategy:
                 with strategy.scope():
                     # Loading might require custom objects or compile=False depending on setup
                     model = keras.saving.load_model(model_path, compile=True)
            else:
                 model = keras.saving.load_model(model_path, compile=True)
            all_models.append(model)

    return all_models


def aggregate_predictions(models, dataset):
    """
    Aggregates predictions from multiple models (e.g., shard models) by averaging.

    Args:
        models: A list of Keras models.
        dataset: A tf.data.Dataset to predict on (should be batched and normalized).

    Returns:
        A numpy array containing the averaged logits or probabilities.
    """
    all_predictions = []
    print(f"Aggregating predictions from {len(models)} models...")
    for model in tqdm(models, desc="Predicting with Shard Models"):
        # Ensure model names are unique if building an ensemble layer later
        # model._name = f"shard_model_{i}" # Optional
        predictions = model.predict(dataset)
        all_predictions.append(predictions)

    # Average the predictions (logits or probabilities)
    avg_predictions = np.mean(all_predictions, axis=0)
    return avg_predictions

# Example of how to create an ensemble model layer (alternative aggregation)
def build_ensemble_model(shard_models, input_shape=(32, 32, 3)):
     """Builds a Keras model that averages the outputs of shard models."""
     model_input = keras.Input(shape=input_shape)
     # Ensure shard models have unique names before this step
     model_outputs = [model(model_input) for model in shard_models]
     ensemble_output = keras.layers.Average()(model_outputs)
     ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output)
     # Compile if needed for evaluation
     # ensemble_model.compile(...)
     return ensemble_model
