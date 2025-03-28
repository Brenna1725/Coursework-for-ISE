import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. Load and prepare the data
def load_and_prepare_data(dataset_path):
    data = pd.read_csv(dataset_path)
    label_col = 'Class-label'  # Change this if needed
    features = data.drop(columns=[label_col])
    labels = data[label_col]
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.3, random_state=42)
    return train_X, test_X, train_y, test_y


# 2. Load the pre-trained model
def load_pretrained_model(model_filepath):
    return load_model(model_filepath)


# 3. Create a pair of test samples (Random Search)
def create_sample_pair(test_data, sensitive_features, nonsensitive_features):
    # Pick a random sample from the test set
    base_sample = test_data.iloc[np.random.choice(len(test_data))]
    perturbed_sample = base_sample.copy()

    # For sensitive features, randomly change the value in the perturbed sample
    for feature in sensitive_features:
        if feature in test_data.columns:
            possible_values = test_data[feature].unique()
            perturbed_sample[feature] = np.random.choice(possible_values)

    # For non-sensitive features, add a small random tweak to both samples
    for feature in nonsensitive_features:
        if feature in test_data.columns:
            min_val = test_data[feature].min()
            max_val = test_data[feature].max()
            tweak = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            base_sample[feature] = np.clip(base_sample[feature] + tweak, min_val, max_val)
            perturbed_sample[feature] = np.clip(perturbed_sample[feature] + tweak, min_val, max_val)

    return base_sample, perturbed_sample


# 4. Assess whether a pair of samples shows discrimination
def assess_discrimination(model, sample1, sample2, threshold=0.05, discr_pairs=None):
    if discr_pairs is None:
        discr_pairs = []  # Initialize list to store pairs that trigger discrimination

    # Convert samples to numpy arrays and reshape for prediction
    sample1_array = np.array(sample1)
    sample2_array = np.array(sample2)

    # Get predictions from the model
    pred1 = model.predict(sample1_array.reshape(1, -1))[0][0]
    pred2 = model.predict(sample2_array.reshape(1, -1))[0][0]

    # If the difference exceeds the threshold, record the pair and flag discrimination
    if abs(pred1 - pred2) > threshold:
        discr_pairs.append((sample1_array, sample2_array))
        return 1
    else:
        return 0


# 5. Compute the Individual Discrimination Instance Ratio (IDI ratio)
def compute_idi_ratio(model, test_data, sensitive_features, nonsensitive_features, num_trials=1000):
    discr_count = 0
    for _ in range(num_trials):
        sample1, sample2 = create_sample_pair(test_data, sensitive_features, nonsensitive_features)
        discr_count += assess_discrimination(model, sample1, sample2)
    idi_ratio = discr_count / num_trials
    return idi_ratio


# 6. Main execution function
def main():
    dataset_path = './dataset/processed_adult.csv'
    model_filepath = './DNN/model_processed_adult.h5'

    train_X, test_X, train_y, test_y = load_and_prepare_data(dataset_path)
    model = load_pretrained_model(model_filepath)

    sensitive_features = ['gender']  # Sensitive features list
    nonsensitive_features = [col for col in test_X.columns if col not in sensitive_features]

    idi_value = compute_idi_ratio(model, test_X, sensitive_features, nonsensitive_features)
    print(f"IDI Ratio: {idi_value}")


if __name__ == "__main__":
    main()
