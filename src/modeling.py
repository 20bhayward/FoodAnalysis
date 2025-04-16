# NewDesign/src/modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA

import models

# Function to prepare data features and targets
def prepare_features_targets(embeddings_df, tagged_df):
    """
    Aligns embeddings and tags based on index.
    Assumes scaling/PCA will be done *after* splitting.
    """
    print(f"\n--- Aligning Features (Embeddings) and Targets (Tags) ---")
    print(f"Input Embeddings shape: {embeddings_df.shape}, Input Tags shape: {tagged_df.shape}")

    common_index = embeddings_df.index.intersection(tagged_df.index)
    if len(common_index) != len(embeddings_df) or len(common_index) != len(tagged_df):
        print("Warning: Indices of embeddings and tags do not perfectly align. Using intersection.")
        embeddings_df = embeddings_df.loc[common_index]
        tagged_df = tagged_df.loc[common_index]
        if len(common_index) == 0:
             print("Error: No common indices found between embeddings and tags after alignment.")
             return None, None, None

    X = embeddings_df.values
    y = tagged_df.values
    # Keep track of the aligned indices for potential later use
    aligned_indices = embeddings_df.index

    print(f"Aligned X shape: {X.shape}, y shape: {y.shape}")
    print("--- Feature and Target Alignment Complete ---")

    return X, y, aligned_indices



def train_neural_network(X_train, X_test, y_train, y_test, tag_columns, batch_size=64, epochs=50):
    """Trains the neural network and evaluates performance."""
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = models.build_neural_network(input_dim, output_dim)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    print("\nNeural Network Architecture:")
    model.summary()

    print(f"Starting NN training for max {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1 # We use 0 for less output, 1 for progress bar, 2 for one line per epoch
    )

    print("Evaluating NN model on test set...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Subset accuracy
    accuracy = accuracy_score(y_test, y_pred)
    hl = hamming_loss(y_test, y_pred)

    tag_f1_scores = {}
    tag_accuracies = {}
    for i, tag in enumerate(tag_columns):
        tag_accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        tag_f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        tag_f1_scores[tag] = tag_f1
        tag_accuracies[tag] = tag_accuracy

    overall_f1_macro = np.mean(list(tag_f1_scores.values())) if tag_f1_scores else 0

    metrics = {
        'model_type': 'Neural Network',
        'accuracy': accuracy,
        'hamming_loss': hl,
        'f1_macro': overall_f1_macro,
        'tag_f1_scores': tag_f1_scores,
        'tag_accuracies': tag_accuracies
    }

    return model, history, metrics


def train_tree_based_models(X_train, X_test, y_train, y_test, tag_columns,
                         optimize_speed=True, sample_size=None):
    """Trains and evaluates RandomForest and optionally GradientBoosting."""
    results = {}

    if optimize_speed and sample_size is not None and sample_size < X_train.shape[0]:
        print(f"\nUsing {sample_size} samples for faster tree model training (out of {X_train.shape[0]})")
        indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    # --- Train Random Forest ---
    print("\nTraining Random Forest model...")
    rf_params = {
        'n_estimators': 50 if optimize_speed else 100,
        'max_depth': 10 if optimize_speed else 20,
        'min_samples_split': 20 if optimize_speed else 10,
        'min_samples_leaf': 10 if optimize_speed else 1,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'bootstrap': True,
        'class_weight': 'balanced' # Added: Can help with imbalanced tags
    }
    rf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
    rf.fit(X_train_sample, y_train_sample)

    rf_pred = rf.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_hl = hamming_loss(y_test, rf_pred)

    rf_tag_f1_scores = {}
    rf_tag_accuracies = {}
    for i, tag in enumerate(tag_columns):
        tag_accuracy = accuracy_score(y_test[:, i], rf_pred[:, i])
        tag_f1 = f1_score(y_test[:, i], rf_pred[:, i], zero_division=0)
        rf_tag_f1_scores[tag] = tag_f1
        rf_tag_accuracies[tag] = tag_accuracy
    rf_overall_f1_macro = np.mean(list(rf_tag_f1_scores.values())) if rf_tag_f1_scores else 0

    rf_metrics = {
        'model_type': 'Random Forest',
        'accuracy': rf_accuracy,
        'hamming_loss': rf_hl,
        'f1_macro': rf_overall_f1_macro,
        'tag_f1_scores': rf_tag_f1_scores,
        'tag_accuracies': rf_tag_accuracies
    }
    results['random_forest'] = (rf, rf_metrics)

    # --- Train Gradient Boosting ---
    if not optimize_speed:
        print("\nTraining Gradient Boosting model...")
        try:
            # Use HistGradientBoosting for hopefully faster training than standard GB
            from sklearn.ensemble import HistGradientBoostingClassifier
            gb_params = {
                'max_iter': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
            gb = MultiOutputClassifier(HistGradientBoostingClassifier(**gb_params), n_jobs=-1)
            # Train on sample or full data based on earlier check
            gb.fit(X_train_sample, y_train_sample)

            gb_pred = gb.predict(X_test)

            gb_accuracy = accuracy_score(y_test, gb_pred)
            gb_hl = hamming_loss(y_test, gb_pred)

            gb_tag_f1_scores = {}
            gb_tag_accuracies = {}
            for i, tag in enumerate(tag_columns):
                tag_accuracy = accuracy_score(y_test[:, i], gb_pred[:, i])
                tag_f1 = f1_score(y_test[:, i], gb_pred[:, i], zero_division=0)
                gb_tag_f1_scores[tag] = tag_f1
                gb_tag_accuracies[tag] = tag_accuracy
            gb_overall_f1_macro = np.mean(list(gb_tag_f1_scores.values())) if gb_tag_f1_scores else 0

            gb_metrics = {
                'model_type': 'Gradient Boosting',
                'accuracy': gb_accuracy,
                'hamming_loss': gb_hl,
                'f1_macro': gb_overall_f1_macro,
                'tag_f1_scores': gb_tag_f1_scores,
                'tag_accuracies': gb_tag_accuracies
            }
            results['gradient_boosting'] = (gb, gb_metrics)
        except ImportError:
            print("HistGradientBoostingClassifier not available (requires scikit-learn >= 0.21). Skipping Gradient Boosting.")
            results['gradient_boosting'] = (None, None)
        except Exception as e:
             print(f"Error during Gradient Boosting training: {e}")
             results['gradient_boosting'] = (None, None)
    else:
        print("\nSkipping Gradient Boosting for faster execution...")
        results['gradient_boosting'] = (None, None)

    return results


def train_regression_model(X_train, X_test, y_train, y_test, tag_columns):
    """Trains and evaluates Logistic Regression."""
    print("\nTraining Logistic Regression model...")
    lr_params = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'solver': 'lbfgs', # Common solver
        'class_weight': 'balanced'
    }
    lr = MultiOutputClassifier(LogisticRegression(**lr_params))
    # Usually train LR on full (scaled) data within the split
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_hl = hamming_loss(y_test, lr_pred)

    lr_tag_f1_scores = {}
    lr_tag_accuracies = {}
    for i, tag in enumerate(tag_columns):
        tag_accuracy = accuracy_score(y_test[:, i], lr_pred[:, i])
        tag_f1 = f1_score(y_test[:, i], lr_pred[:, i], zero_division=0)
        lr_tag_f1_scores[tag] = tag_f1
        lr_tag_accuracies[tag] = tag_accuracy
    lr_overall_f1_macro = np.mean(list(lr_tag_f1_scores.values())) if lr_tag_f1_scores else 0

    lr_metrics = {
        'model_type': 'Logistic Regression',
        'accuracy': lr_accuracy,
        'hamming_loss': lr_hl,
        'f1_macro': lr_overall_f1_macro,
        'tag_f1_scores': lr_tag_f1_scores,
        'tag_accuracies': lr_tag_accuracies
    }

    return lr, lr_metrics