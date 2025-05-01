import tensorflow as tf
from tensorflow import keras
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

def normalize(image, label, denorm=False):
    """Normalizes CIFAR-10 images or denormalizes them.

    Args:
        image (tf.Tensor): Input image tensor.
        label (tf.Tensor): Input label tensor.
        denorm (bool): If True, perform denormalization. Defaults to False.

    Returns:
        tuple: A tuple containing the normalized/denormalized image and the label.
    """
    rescale = keras.layers.Rescaling(scale=1./255.)
    # CIFAR-10 normalization constants
    mean = [0.4914, 0.4822, 0.4465]
    variance = [np.square(0.2023), np.square(0.1994), np.square(0.2010)]

    norms = keras.layers.Normalization(
        mean=mean,
        variance=variance,
        invert=denorm,
        axis=-1,
    )

    if not denorm:
        image = rescale(image) # Rescale to [0, 1] before normalizing
    return norms(image), label

def load_cifar10(batch_size=128, forget_fraction=0.1, seed=0):
    """Loads and prepares the CIFAR-10 dataset.

    Splits the data into training, validation, test, forget, and retain sets.
    Applies normalization and batching.

    Args:
        batch_size (int): The batch size for the datasets. Defaults to 128.
        forget_fraction (float): The fraction of the training set to designate as
                                 the forget set. Defaults to 0.1.
        seed (int): Random seed for dataset splitting. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - train_ds (tf.data.Dataset): Full training dataset.
            - val_ds (tf.data.Dataset): Validation dataset.
            - test_ds (tf.data.Dataset): Test dataset.
            - forget_ds (tf.data.Dataset): Forget dataset (subset of training).
            - retain_ds (tf.data.Dataset): Retain dataset (training - forget).
            - forget_set_unbatched (tf.data.Dataset): Unbatched forget dataset.
            - retain_set_unbatched (tf.data.Dataset): Unbatched retain dataset.
            - test_set_unbatched (tf.data.Dataset): Unbatched test dataset.
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), held_out = keras.datasets.cifar10.load_data()
    print(f"Original training data shape: {x_train.shape}")
    print(f"Original held-out data shape: {held_out[0].shape}")

    # Split held-out data into validation and test sets (80% validation, 20% test)
    # Note: The original notebook used left_size=0.2 for the *test* set from held_out.
    # Adjusting here for clarity: 80% validation, 20% test.
    val_set_unbatched, test_set_unbatched = keras.utils.split_dataset(
        held_out, left_size=0.8, seed=seed
    )
    print(f"Validation set size: {val_set_unbatched.cardinality().numpy()}")
    print(f"Test set size: {test_set_unbatched.cardinality().numpy()}")


    # Create the full training dataset pipeline
    train_ds_unbatched = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=8 * batch_size) # Shuffle before batching
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    print(f"Full training dataset size: {train_ds_unbatched.cardinality().numpy()}")

    # Create validation and test dataset pipelines
    val_ds = val_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    test_ds_normalized = test_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds_normalized.batch(batch_size).prefetch(AUTOTUNE)


    # Split the unbatched training set into forget and retain sets
    forget_set_unbatched, retain_set_unbatched = keras.utils.split_dataset(
        train_ds_unbatched, left_size=forget_fraction, seed=seed
    )
    print(f"Forget set size: {forget_set_unbatched.cardinality().numpy()}")
    print(f"Retain set size: {retain_set_unbatched.cardinality().numpy()}")

    # Create forget and retain dataset pipelines
    forget_ds = forget_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    forget_ds = forget_ds.batch(batch_size).prefetch(AUTOTUNE)

    retain_ds = retain_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    retain_ds = retain_ds.shuffle(buffer_size=8 * batch_size) # Shuffle retain set for training
    retain_ds = retain_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Also return unbatched, normalized versions for MIA
    forget_set_normalized = forget_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)
    retain_set_normalized = retain_set_unbatched.map(normalize, num_parallel_calls=AUTOTUNE)


    print("CIFAR-10 datasets prepared.")
    return (train_ds, val_ds, test_ds, forget_ds, retain_ds,
            forget_set_normalized, retain_set_normalized, test_ds_normalized)

def get_dataset_partitions(dataset, num_shards):
    """Partitions a tf.data.Dataset into a specified number of shards.

    Args:
        dataset (tf.data.Dataset): The dataset to partition (should be unbatched).
        num_shards (int): The number of partitions to create.

    Returns:
        list: A list of tf.data.Dataset objects, each representing a shard.
    """
    shards = []
    dataset_size = dataset.cardinality().numpy()
    if dataset_size < num_shards:
        raise ValueError("Number of shards cannot be greater than the dataset size.")

    print(f"Partitioning dataset of size {dataset_size} into {num_shards} shards...")
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shards.append(shard)
        print(f"  Shard {i} size: {shard.cardinality().numpy()}")
    return shards
