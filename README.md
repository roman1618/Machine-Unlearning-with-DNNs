# Machine Unlearning with DNNs

This repository explores various machine unlearning techniques for Deep Neural Networks (DNNs), focusing on image classification tasks with CIFAR-10 using ResNet18.

## Project Structure

```
.
├── data/             # Data loading and preprocessing (CIFAR-10)
├── evaluation/       # Evaluation metrics (accuracy, loss) and Membership Inference Attacks (MIA)
├── examples/         # Example Jupyter notebooks demonstrating usage of the modules
├── models/           # Model definitions (ResNet18)
├── training/         # Training utilities (placeholder)
├── unlearning/       # Implementations of unlearning techniques
│   ├── combined.py
│   ├── fragmentation.py # SISA/SNISA logic (potentially)
│   ├── layer_reset.py   # Layer reinitialization and confusion (UnconLa)
│   └── weight_pruning.py
├── utils/            # Helper functions (device setup, model cloning, plotting)
├── weights_and_sim/  # Weight similarity analysis (notebooks)
├── *.ipynb           # Original research notebooks (may not use the modular structure)
├── *.keras           # Saved model weights (e.g., resnet18_cifar10.keras)
└── README.md         # This file
```

## Implemented Techniques

This codebase includes modules related to:

*   **Baseline Training:** Standard training of a ResNet18 model on CIFAR-10.
*   **SNISA (Sharded, Non-Isolated, Sliced, Aggregated):** Training an ensemble of models on different partitions (shards) of the data. Unlike SISA (Isolated), SNISA allows shards to potentially overlap or influence each other depending on the specific implementation details (though the current example uses distinct shards).
*   **UnconLa (Unlearning by Confusing Layers):** Adding noise to model weights periodically during training as part of the unlearning process.
*   **Combined Approaches:** Integrating techniques like SNISA and UnconLa (demonstrated in the example).
*   **Evaluation:** Assessing model accuracy and unlearning effectiveness using standard metrics and Membership Inference Attacks (MIA).

## Getting Started

1.  **Environment Setup:** Ensure you have Python and TensorFlow installed. You might need other libraries like `scikit-learn`, `matplotlib`, `numpy`, `keras-cv`.
    ```bash
    pip install tensorflow scikit-learn matplotlib numpy keras-cv tqdm
    ```
2.  **Pre-trained Model:** Some examples require a pre-trained base model (`resnet18_cifar10.keras` in the root directory). If not present, you may need to train one first (check example notebooks or the original `Base_model.ipynb` for guidance).
3.  **Run Examples:** Explore the Jupyter notebooks in the `examples/` directory to see how to use the different modules and reproduce experiments.
    *   `snisa_unconla_example.ipynb`: Demonstrates the combination of SNISA and UnconLa.

## Usage

Import functions from the respective modules (`data`, `models`, `unlearning`, `evaluation`, `utils`) into your Python scripts or notebooks.

```python
# Example imports
from utils.setup import set_device
from data.cifar10 import load_cifar10
from models.resnet18 import build_resnet18_model
from evaluation.mia import perform_mia_on_forget_test
from unlearning.layer_reset import vision_confuser
# ... and so on
```

Refer to the example notebooks for detailed usage patterns.
