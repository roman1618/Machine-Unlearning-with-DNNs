import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import os
import warnings

def suppress_tf_warnings():
    """Suppresses TensorFlow warnings and logs."""
    warnings.simplefilter(action="ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def set_device(mixed_precision=True, set_jit=False):
    """Configures TensorFlow device usage (GPU/CPU), mixed precision, and JIT.

    Args:
        mixed_precision (bool): Whether to enable mixed precision (mixed_float16).
                                Defaults to True.
        set_jit (bool): Whether to enable XLA JIT compilation. Defaults to False.

    Returns:
        tuple: A tuple containing the TensorFlow distribution strategy
               and the list of physical devices detected.
    """
    suppress_tf_warnings()
    # Print out the detected devices
    list_ld = device_lib.list_local_devices()
    print("Detected devices:")
    for dev in list_ld:
        print(f"- {dev.name} (Memory Limit: {dev.memory_limit} bytes)")

    # Get the list of physical devices
    physical_devices = tf.config.list_physical_devices(
        'GPU' if len(list_ld) > 1 else 'CPU' # Assumes at least one CPU device
    )
    print(f"Using device type: {'GPU' if 'GPU' in physical_devices[-1].name else 'CPU'}")

    # For GPU devices, configure related stuff
    if 'GPU' in physical_devices[-1].name:
        tf.config.optimizer.set_jit(set_jit)
        print(f"JIT Compilation: {'Enabled' if set_jit else 'Disabled'}")
        if mixed_precision:
            keras.mixed_precision.set_global_policy(
                "mixed_float16"
            )
            print("Mixed Precision: Enabled (mixed_float16)")
        else:
            keras.mixed_precision.set_global_policy(
                keras.backend.floatx()
            )
            print(f"Mixed Precision: Disabled (using {keras.backend.floatx()})")
        # Enable memory growth for GPUs to avoid allocating all memory at once
        for pd in physical_devices:
            if pd.device_type == 'GPU':
                 try:
                    tf.config.experimental.set_memory_growth(pd, True)
                    print(f"Memory growth enabled for {pd.name}")
                 except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"Could not set memory growth for {pd.name}: {e}")

    # Use MirroredStrategy for multi-GPU training if available, otherwise default strategy
    if len(physical_devices) > 1 and any(dev.device_type == 'GPU' for dev in physical_devices):
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with devices: {strategy.extended.worker_devices}")
    else:
        strategy = tf.distribute.get_strategy() # Default strategy
        print("Using default TensorFlow strategy.")

    return strategy, physical_devices
