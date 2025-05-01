import tensorflow as tf
from tensorflow import keras
import time

def train_model(model, train_ds, val_ds, epochs, callbacks=None, verbose=1):
    """Trains a Keras model using the provided datasets.

    Args:
        model (tf.keras.Model): The Keras model to train.
        train_ds (tf.data.Dataset): The training dataset (batched).
        val_ds (tf.data.Dataset): The validation dataset (batched).
        epochs (int): The number of epochs to train for.
        callbacks (list, optional): List of Keras callbacks to use during training.
                                    Defaults to None.
        verbose (int): Verbosity mode for model.fit (0, 1, or 2). Defaults to 1.

    Returns:
        tf.keras.callbacks.History: The history object returned by model.fit.
    """
    print(f"Starting training for model '{model.name}' for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=verbose
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    return history
