import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from keras_cv.models import ResNet18Backbone

def build_resnet18_model(strategy, input_shape=(32, 32, 3), num_classes=10, model_name="resnet18_cifar10"):
    """Builds a ResNet18 model for CIFAR-10 classification.

    Args:
        strategy (tf.distribute.Strategy): The distribution strategy to use.
        input_shape (tuple): The shape of the input images. Defaults to (32, 32, 3).
        num_classes (int): The number of output classes. Defaults to 10.
        model_name (str): The name for the Keras model. Defaults to "resnet18_cifar10".

    Returns:
        tf.keras.Model: The compiled ResNet18 Keras model.
    """
    with strategy.scope():
        model = keras.Sequential(
            [
                # Note: include_rescaling=False because normalization is handled in the dataset pipeline
                ResNet18Backbone(include_rescaling=False, input_shape=input_shape),
                layers.GlobalMaxPooling2D(),
                # Output layer with float32 dtype for mixed precision compatibility
                layers.Dense(num_classes, activation="softmax", dtype=tf.float32),
            ],
            name=model_name
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')],
            jit_compile=True # Enable JIT compilation
        )
        print(f"Compiled ResNet18 model '{model_name}' within strategy scope.")
    return model
