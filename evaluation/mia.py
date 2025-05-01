import numpy as np
from sklearn import linear_model, model_selection
from .metrics import compute_losses # Import from sibling module
import tensorflow as tf

def simple_mia(sample_losses, member_labels, n_splits=10, random_state=0):
    """Computes cross-validation score of a simple membership inference attack.

    Trains a logistic regression model to distinguish between members (label 1)
    and non-members (label 0) based on their loss values.

    Args:
        sample_losses (np.ndarray): Array of loss values for each sample.
                                    Should be shape (n_samples,).
        member_labels (np.ndarray): Array of labels indicating membership (1 for member,
                                    0 for non-member). Should be shape (n_samples,).
        n_splits (int): Number of splits for cross-validation. Defaults to 10.
        random_state (int): Random seed for splitting. Defaults to 0.

    Returns:
        np.ndarray: An array containing the accuracy score for each cross-validation split.
    """
    unique_members = np.unique(member_labels)
    if not np.all(np.isin(unique_members, [0, 1])):
        raise ValueError("member_labels should only contain 0s and 1s")
    if sample_losses.ndim != 1 or sample_losses.shape[0] != member_labels.shape[0]:
         raise ValueError("sample_losses must be a 1D array matching the size of member_labels.")


    print(f"Performing Simple MIA with {n_splits} splits...")
    attack_model = linear_model.LogisticRegression(solver='liblinear') # Added solver for potential convergence warnings
    # Reshape losses to (n_samples, 1) as expected by scikit-learn
    sample_losses_reshaped = sample_losses.reshape(-1, 1)

    # Use StratifiedShuffleSplit for potentially imbalanced datasets
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )

    try:
        scores = model_selection.cross_val_score(
            attack_model, sample_losses_reshaped, member_labels, cv=cv, scoring="accuracy"
        )
        print(f"MIA Cross-Validation Accuracy (mean): {scores.mean():.3f}")
        print(f"MIA Cross-Validation Accuracy (std): {scores.std():.3f}")
        return scores
    except Exception as e:
        print(f"Error during MIA cross-validation: {e}")
        # Handle potential issues, e.g., if classes are not represented in a split
        return np.array([np.nan] * n_splits) # Return NaNs or handle appropriately


def perform_mia_on_forget_test(model, forget_set_normalized, test_set_normalized, batch_size=128):
    """Performs the simple MIA comparing the forget set and the test set.

    Args:
        model (tf.keras.Model): The model to evaluate.
        forget_set_normalized (tf.data.Dataset): The normalized forget dataset (unbatched).
        test_set_normalized (tf.data.Dataset): The normalized test dataset (unbatched).
        batch_size (int): Batch size for computing losses.

    Returns:
        np.ndarray: The MIA scores from cross-validation. Returns None if an error occurs.
    """
    print("Performing MIA between Forget Set and Test Set...")

    # Batch the datasets for efficient loss computation
    forget_ds_batched = forget_set_normalized.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds_batched = test_set_normalized.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compute losses
    try:
        forget_losses = compute_losses(model, forget_ds_batched)
        test_losses = compute_losses(model, test_ds_batched)
    except Exception as e:
        print(f"Error computing losses for MIA: {e}")
        return None

    # Balance the datasets for MIA: Sub-sample the larger set
    n_forget = len(forget_losses)
    n_test = len(test_losses)
    print(f"Forget set losses calculated: {n_forget}")
    print(f"Test set losses calculated: {n_test}")


    if n_forget == 0 or n_test == 0:
        print("Error: Cannot perform MIA with zero samples in forget or test set.")
        return None

    min_len = min(n_forget, n_test)
    print(f"Balancing for MIA using {min_len} samples from each set.")

    # Ensure reproducibility if needed when shuffling
    np.random.seed(0) # Use a fixed seed or pass as argument if needed

    if n_forget > min_len:
        forget_indices = np.random.choice(n_forget, min_len, replace=False)
        balanced_forget_losses = forget_losses[forget_indices]
    else:
        balanced_forget_losses = forget_losses

    if n_test > min_len:
        test_indices = np.random.choice(n_test, min_len, replace=False)
        balanced_test_losses = test_losses[test_indices]
    else:
        balanced_test_losses = test_losses


    # Prepare data for simple_mia function
    # Losses: Concatenate balanced test losses (label 0) and forget losses (label 1)
    # Labels: Create corresponding 0s and 1s
    mia_sample_losses = np.concatenate((balanced_test_losses, balanced_forget_losses))
    mia_member_labels = np.array([0] * min_len + [1] * min_len)


    # Perform the MIA
    mia_scores = simple_mia(mia_sample_losses, mia_member_labels)

    return mia_scores

