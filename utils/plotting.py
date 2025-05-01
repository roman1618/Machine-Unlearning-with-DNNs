import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_history(history, title="Model Training History"):
    """Plots the training and validation loss and accuracy from a Keras history object.

    Args:
        history (tf.keras.callbacks.History or dict): Keras history object or dictionary
                                                     containing 'loss', 'accuracy',
                                                     'val_loss', 'val_accuracy'.
        title (str): The title for the plot.
    """
    if isinstance(history, dict):
        history_df = pd.DataFrame(history)
    elif hasattr(history, 'history'):
         history_df = pd.DataFrame(history.history)
    else:
        raise ValueError("Input must be a Keras History object or a dictionary.")

    print(f"Plotting training history: {title}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)

    # Plot Loss
    axes[0].plot(history_df[['loss', 'val_loss']])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend(['Train', 'Validation'])
    # axes[0].set_ylim(bottom=0) # Optional: set y-axis lower limit

    # Plot Accuracy
    axes[1].plot(history_df[['accuracy', 'val_accuracy']])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)
    axes[1].legend(['Train', 'Validation'])
    axes[1].set_ylim(0, 1) # Accuracy is between 0 and 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


def plot_loss_distribution(losses1, losses2, label1, label2, title="Loss Distribution", xlim=None):
    """Plots histograms of two sets of losses.

    Args:
        losses1 (np.ndarray): First array of loss values.
        losses2 (np.ndarray): Second array of loss values.
        label1 (str): Label for the first set of losses.
        label2 (str): Label for the second set of losses.
        title (str): Title for the plot.
        xlim (tuple, optional): Tuple specifying the x-axis limits (min, max). Defaults to None.
    """
    print(f"Plotting loss distribution: {title}")
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.hist(losses1, density=True, alpha=0.6, bins=50, label=label1)
    plt.hist(losses2, density=True, alpha=0.6, bins=50, label=label2)
    plt.xlabel("Loss", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    if xlim:
        plt.xlim(xlim)
    else:
        # Determine reasonable xlim if not provided, e.g., based on combined data
        combined_losses = np.concatenate((losses1, losses2))
        # Avoid extreme outliers if necessary, e.g., using percentiles
        q99 = np.percentile(combined_losses, 99)
        plt.xlim((0, q99))


    plt.yscale("log") # Use log scale for better visibility
    plt.legend(frameon=False, fontsize=12)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_mia_comparison(mia_scores_before, mia_scores_after, title1="Before Unlearning", title2="After Unlearning", losses_before=None, losses_after=None):
    """Compares loss distributions and MIA scores before and after unlearning.

    Args:
        mia_scores_before (np.ndarray): MIA scores before unlearning.
        mia_scores_after (np.ndarray): MIA scores after unlearning.
        title1 (str): Subplot title for the 'before' state.
        title2 (str): Subplot title for the 'after' state.
        losses_before (tuple, optional): Tuple of (test_losses, forget_losses) before unlearning.
        losses_after (tuple, optional): Tuple of (test_losses, forget_losses) after unlearning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True) # Share y-axis for comparison
    fig.suptitle("Membership Inference Attack Comparison", fontsize=16)

    mean_mia_before = np.mean(mia_scores_before) if mia_scores_before is not None else np.nan
    mean_mia_after = np.mean(mia_scores_after) if mia_scores_after is not None else np.nan


    # Plot Before Unlearning
    axes[0].set_title(f"{title1}\nAttack Accuracy: {mean_mia_before:.3f}")
    if losses_before:
        test_losses, forget_losses = losses_before
        axes[0].hist(test_losses, density=True, alpha=0.6, bins=50, label="Test Set")
        axes[0].hist(forget_losses, density=True, alpha=0.6, bins=50, label="Forget Set")
        axes[0].set_yscale("log")
        # Determine shared xlim based on combined 'before' losses
        combined_losses_before = np.concatenate((test_losses, forget_losses))
        q99_before = np.percentile(combined_losses_before, 99)
        xlim_max = q99_before
    else:
        axes[0].text(0.5, 0.5, 'Loss data not provided', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)


    # Plot After Unlearning
    axes[1].set_title(f"{title2}\nAttack Accuracy: {mean_mia_after:.3f}")
    if losses_after:
        test_losses, forget_losses = losses_after
        axes[1].hist(test_losses, density=True, alpha=0.6, bins=50, label="Test Set")
        axes[1].hist(forget_losses, density=True, alpha=0.6, bins=50, label="Forget Set")
        axes[1].set_yscale("log")
        # Use the same xlim as the 'before' plot if available
        if losses_before:
             axes[0].set_xlim((0, xlim_max))
             axes[1].set_xlim((0, xlim_max))

    else:
         axes[1].text(0.5, 0.5, 'Loss data not provided', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    # Common Formatting
    for ax in axes:
        ax.set_xlabel("Loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=12)
    axes[1].legend(frameon=False, fontsize=12)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Add more specific plotting functions as needed (e.g., for fragmentation results)
