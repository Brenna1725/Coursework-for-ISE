import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.model_selection import train_test_split

# 1. Load and preprocess the data (this part stays the same)
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    target_column = 'Class-label'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


# 2. Load the pre-trained model (using keras_load_model directly to avoid name conflicts)
def load_pretrained_model(model_path):
    return keras_load_model(model_path)


# 3. Use tabu search to generate a pair of samples
def tabu_search_sample_pair(X_test, sensitive_columns, non_sensitive_columns, model, threshold=0.05, max_iter=50, tabu_size=10):
    """
    Use tabu search to find a pair of samples that maximizes the difference in model predictions.
    Returns 1 if the difference is above the threshold, otherwise 0.
    """
    # Pick a random sample from X_test and duplicate it to form a pair.
    current_a = X_test.iloc[np.random.choice(len(X_test))].copy()
    current_b = current_a.copy()

    # For sensitive features, randomly change values in sample B.
    for col in sensitive_columns:
        if col in X_test.columns:
            unique_values = X_test[col].unique()
            current_b[col] = np.random.choice(unique_values)

    # For non-sensitive features, add a small random tweak to both samples.
    for col in non_sensitive_columns:
        if col in X_test.columns:
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            current_a[col] = np.clip(current_a[col] + perturbation, min_val, max_val)
            current_b[col] = np.clip(current_b[col] + perturbation, min_val, max_val)

    # Objective: get the absolute difference between model predictions.
    def evaluate_pair(sample_a, sample_b):
        arr_a = np.array(sample_a)
        arr_b = np.array(sample_b)
        pred_a = model.predict(arr_a.reshape(1, -1))[0][0]
        pred_b = model.predict(arr_b.reshape(1, -1))[0][0]
        return abs(pred_a - pred_b)

    current_diff = evaluate_pair(current_a, current_b)
    best_a, best_b, best_diff = current_a.copy(), current_b.copy(), current_diff

    # Tabu list to store already-visited sample pairs (using a simple hash)
    tabu_list = []

    def hash_pair(a, b):
        return str(tuple(a)) + "|" + str(tuple(b))

    tabu_list.append(hash_pair(current_a, current_b))

    # Main loop for tabu search.
    for _ in range(max_iter):
        neighbors = []

        # Try small tweaks on non-sensitive features.
        for col in non_sensitive_columns:
            neighbor_a = current_a.copy()
            neighbor_b = current_b.copy()
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.05 * (max_val - min_val), 0.05 * (max_val - min_val))
            neighbor_a[col] = np.clip(neighbor_a[col] + perturbation, min_val, max_val)
            neighbor_b[col] = np.clip(neighbor_b[col] + perturbation, min_val, max_val)
            neighbors.append((neighbor_a, neighbor_b))

        # Also, try changing the sensitive features (only in sample B).
        for col in sensitive_columns:
            neighbor_a = current_a.copy()
            neighbor_b = current_b.copy()
            unique_values = X_test[col].unique()
            neighbor_b[col] = np.random.choice(unique_values)
            neighbors.append((neighbor_a, neighbor_b))

        # Pick the neighbor with the biggest difference that isn't in the tabu list.
        candidate_found = False
        candidate_a, candidate_b = None, None
        candidate_diff = -1

        for cand_a, cand_b in neighbors:
            key = hash_pair(cand_a, cand_b)
            if key in tabu_list:
                continue
            diff = evaluate_pair(cand_a, cand_b)
            if diff > candidate_diff:
                candidate_diff = diff
                candidate_a, candidate_b = cand_a.copy(), cand_b.copy()
                candidate_found = True

        # Update current solution and tabu list if a candidate was found.
        if candidate_found:
            current_a, current_b = candidate_a.copy(), candidate_b.copy()
            tabu_list.append(hash_pair(current_a, current_b))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
            if candidate_diff > best_diff:
                best_diff = candidate_diff
                best_a, best_b = current_a.copy(), current_b.copy()
        else:
            break  # No valid candidate found, so stop.

    # Return the best pair and a flag (1 if discrimination is detected, else 0).
    return best_a, best_b, 1 if best_diff > threshold else 0


# 4. Calculate the individual discrimination instance ratio using tabu search.
def calculate_idi_ratio_tabu(model, X_test, sensitive_columns, non_sensitive_columns, num_iterations=100):
    discrimination_count = 0
    for _ in range(num_iterations):
        sample_a, sample_b, indicator = tabu_search_sample_pair(
            X_test, sensitive_columns, non_sensitive_columns, model
        )
        discrimination_count += indicator
    return discrimination_count / num_iterations


# 5. Main function: load data, model, set sensitive features, and compute the IDI ratio.
def main():
    file_path = './dataset/processed_adult.csv'
    model_path = './DNN/model_processed_adult.h5'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    model = load_pretrained_model(model_path)

    sensitive_columns = ['age']  # Change this list if needed.
    non_sensitive_columns = [col for col in X_test.columns if col not in sensitive_columns]

    idi_ratio = calculate_idi_ratio_tabu(model, X_test, sensitive_columns, non_sensitive_columns)
    print(f"IDI Ratio (Tabu Search): {idi_ratio}")


if __name__ == "__main__":
    main()
