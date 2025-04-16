# NewDesign/src/pipelines.py

import os
import pandas as pd
import numpy as np
import torch
import random
import pickle
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.metrics import hamming_loss
import traceback

from torch.utils.data import DataLoader, SequentialSampler

# Attempt SBERT import, handle if missing
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Set to None if library not installed
    SentenceTransformer = None

# Direct imports from other src modules
import preprocessing
import embeddings
import modeling
import visualization
import lstm_autoencoder
import models


# Helper function for error analysis
def print_error_analysis(y_true, y_pred, original_indices, processed_df, tag_columns, n_examples=5):
    """Prints details for samples with the highest Hamming distance."""
    try:
        # Calculate Hamming distance (number of incorrect labels per sample)
        hamming_dist = np.sum(np.abs(y_true.astype(int) - y_pred.astype(int)), axis=1)
        # Get indices sorted by distance descending (worst first)
        worst_indices_local = np.argsort(hamming_dist)[::-1]

        print(f"\n--- Top {min(n_examples, len(worst_indices_local))} Misclassified Examples (by Hamming Distance) ---")

        # Prepare lookup dataframe (use ID index if available and unique)
        has_id_col = 'id' in processed_df.columns
        processed_df_lookup = processed_df
        use_id_lookup = False
        if has_id_col and 'id' in processed_df.index.names: # Check if 'id' is already the index
             if processed_df.index.is_unique:
                  processed_df_lookup = processed_df # Already indexed and unique
                  use_id_lookup = True
                  print("Using unique 'id' index from processed_df for lookup.")
             else:
                  print("Warning: 'id' index in processed_df is not unique. Lookup might be ambiguous.")
                  processed_df_lookup = processed_df
                  use_id_lookup = True # Still use ID index, but warn
        elif has_id_col and processed_df['id'].is_unique:
             processed_df_lookup = processed_df.set_index('id', drop=False)
             use_id_lookup = True
             print("Using unique 'id' column (setting as index) for lookup.")
        else:
            print("Warning: Using default index for error analysis lookup (ID column missing, not index, or not unique).")


        for i in range(min(n_examples, len(worst_indices_local))):
            local_idx = worst_indices_local[i] # Index within the current test/validation split
            try:
                 # Map local index back to the original index/ID value
                 original_idx_val = original_indices[local_idx]
            except IndexError:
                 print(f"Error accessing original_indices at local index {local_idx}. Skipping example.")
                 continue

            dist = hamming_dist[local_idx]
            text = "Text Not Found"

            try:
                 # Use .loc for lookup on the prepared dataframe
                 lookup_result = processed_df_lookup.loc[original_idx_val]
                 # Handle case where lookup returns multiple rows (non-unique index)
                 if isinstance(lookup_result, pd.DataFrame):
                     text = lookup_result['cleaned_text_for_lstm'].iloc[0]
                 else: # It's a Series (unique index)
                     text = lookup_result['cleaned_text_for_lstm']
            except (KeyError, IndexError, TypeError) as e:
                 text = f"Text Not Found (Lookup Error for index/ID {original_idx_val}: {e})"

            # Find indices of true and predicted tags
            true_tags_indices = np.where(y_true[local_idx] == 1)[0]
            pred_tags_indices = np.where(y_pred[local_idx] == 1)[0]

            # Map indices to tag names
            true_tags = [tag_columns[j] for j in true_tags_indices if j < len(tag_columns)]
            pred_tags = [tag_columns[j] for j in pred_tags_indices if j < len(tag_columns)]

            print(f"\nExample {i+1} (Original Index/ID: {original_idx_val}, Hamming Distance: {dist})")
            print(f"  Text: {text[:150]}...")
            print(f"  True Tags: {true_tags if true_tags else ['None']}")
            print(f"  Predicted Tags: {pred_tags if pred_tags else ['None']}")

    except Exception as e:
        print(f"Error during error analysis: {e}")
        traceback.print_exc()
    print("--- End Error Analysis ---")


def run_classification_pipeline(
    processed_data_path="data/processed/processed_recipes.csv",
    embedding_output_dir="results",
    embedding_checkpoint_dir="checkpoints/lstm_classifier",
    embedding_input_path=None, # Path to pre-generated embeddings (.parquet)
    viz_output_dir="results/visualizations",
    classification_viz_dir="results/visualizations/classification",
    embedding_tags_viz_dir="results/visualizations/embedding_tags_lstm",
    fast_mode=True,
    run_preprocessing=False,
    raw_recipes_path="data/raw/RAW_recipes.csv",
    raw_interactions_path="data/raw/RAW_interactions.csv",
    k_folds=1, # 1 means single train/test split, > 1 means K-Fold CV
    random_state=42,
    n_top_tags_override=None # Allows overriding N_TOP_TAGS from preprocessing.py for ablation
    ):
    """
    Runs the complete multi-label tag classification pipeline.
    Handles preprocessing, embedding generation (LSTM default) or loading,
    model training (NN, RF, GB, LR), evaluation (single split or K-Fold),
    visualization, and optional tag ablation studies.
    """
    print("Starting Classification Pipeline...")
    print(f"Fast mode: {fast_mode}")
    if k_folds > 1: print(f"K-Fold CV enabled with k={k_folds}")
    if n_top_tags_override: print(f"Overriding N_TOP_TAGS to: {n_top_tags_override}")

    # --- 1. Load or Generate Processed Data ---
    if run_preprocessing:
        print(f"\nStep 1a: Running Preprocessing...")
        original_n_tags_setting = preprocessing.N_TOP_TAGS
        if n_top_tags_override is not None:
            print(f"   Temporarily setting preprocessing.N_TOP_TAGS = {n_top_tags_override}")
            preprocessing.N_TOP_TAGS = n_top_tags_override

        try:
            processed_df = preprocessing.preprocess_data(
                raw_recipes_path=raw_recipes_path,
                raw_interactions_path=raw_interactions_path,
                output_path=processed_data_path
            )
        finally:
            # Always restore the original setting
            preprocessing.N_TOP_TAGS = original_n_tags_setting
            print(f"   Restored preprocessing.N_TOP_TAGS = {original_n_tags_setting}")

        if processed_df is None: return None
    else:
        print(f"\nStep 1a: Loading Preprocessed Data from {processed_data_path}...")
        try:
            processed_df = pd.read_csv(processed_data_path)
            print(f"Loaded {len(processed_df)} processed recipes.")
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {processed_data_path}.")
            print("Consider running with --run-preprocessing flag.")
            return None
        except Exception as e: print(f"Error loading processed data: {e}"); return None

    # Establish original index mapping (ID if present and unique, otherwise RangeIndex)
    original_id_column = None
    using_id_col_ref = False
    if 'id' in processed_df.columns:
         original_id_column = processed_df['id'].copy()
         if original_id_column.is_unique:
             using_id_col_ref = True
             print("Using unique 'id' column as original index reference.")
             # Set ID as index for easier lookups later, keep 'id' column too
             processed_df = processed_df.set_index('id', drop=False)
         else:
             print("Warning: Loaded 'id' column is not unique! Using default RangeIndex reference.")
             processed_df = processed_df.reset_index(drop=True) # Ensure clean RangeIndex
             original_id_column = pd.RangeIndex(start=0, stop=len(processed_df), step=1)
    else:
         print("Warning: 'id' column missing from processed data. Using default RangeIndex reference.")
         processed_df = processed_df.reset_index(drop=True) # Ensure clean RangeIndex
         original_id_column = pd.RangeIndex(start=0, stop=len(processed_df), step=1)

    if 'cleaned_text_for_lstm' not in processed_df.columns: print("Error: 'cleaned_text_for_lstm' column not found."); return None
    tag_columns = processed_df.columns[processed_df.columns.str.contains('tag_', case=False)].tolist()
    if not tag_columns: print("Error: No 'tag_' columns found in loaded/processed data."); return None

    actual_n_tags = len(tag_columns)
    print(f"Using {actual_n_tags} tag columns for prediction: {tag_columns[:5]}..." if actual_n_tags > 5 else f"Using {actual_n_tags} tag columns: {tag_columns}")
    if n_top_tags_override is not None and actual_n_tags != n_top_tags_override:
         print(f"Warning: Loaded/Processed data has {actual_n_tags} tags, but override requested {n_top_tags_override}. Proceeding with {actual_n_tags} tags found.")


    # --- 2. Generate or Load Embeddings ---
    if embedding_input_path:
        print(f"\nStep 1b: Loading Pre-generated Embeddings from {embedding_input_path}...")
        try:
            embeddings_df = pd.read_parquet(embedding_input_path)
            print(f"Loaded embeddings shape: {embeddings_df.shape}")

            # Align embeddings with processed_df based on index
            common_index = processed_df.index.intersection(embeddings_df.index)
            if len(common_index) == 0:
                raise ValueError("No common indices found between loaded embeddings and processed data.")
            if len(common_index) < len(processed_df) or len(common_index) < len(embeddings_df):
                print(f"Warning: Aligning data on {len(common_index)} common indices.")

            processed_df = processed_df.loc[common_index]
            embeddings_df = embeddings_df.loc[common_index]
            original_id_column = original_id_column.loc[common_index] # Keep original IDs aligned

            print(f"Aligned data shape: {processed_df.shape}, Embeddings shape: {embeddings_df.shape}")
            if len(processed_df) == 0:
                 raise ValueError("Alignment resulted in zero samples.")

        except FileNotFoundError:
            print(f"Error: Embeddings file not found at {embedding_input_path}.")
            return None
        except Exception as e:
            print(f"Error loading or aligning embeddings file: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"\nStep 1b: Generating/Loading Default LSTM Embeddings...")
        text_df_for_embedding = processed_df[['cleaned_text_for_lstm']].copy()

        embeddings_df = embeddings.create_and_apply_embeddings(
            df=text_df_for_embedding, # Pass df with correct index
            text_column='cleaned_text_for_lstm',
            subsample_size=50000 if not fast_mode else 10000, # Larger sample if not fast mode
            num_epochs=10 if not fast_mode else 5,
            checkpoint_dir=embedding_checkpoint_dir, output_dir=embedding_output_dir,
            random_state=random_state
        )
        if embeddings_df is None: print("Embedding generation failed."); return None
        # Re-align after generation, just in case index got lost (shouldn't happen ideally)
        common_index = processed_df.index.intersection(embeddings_df.index)
        if len(common_index) != len(processed_df) or len(common_index) != len(embeddings_df):
             print("Warning: Index mismatch after default embedding generation. Re-aligning.")
             processed_df = processed_df.loc[common_index]
             embeddings_df = embeddings_df.loc[common_index]
             original_id_column = original_id_column.loc[common_index]
        print(f"Generated embeddings shape: {embeddings_df.shape}")


    # --- 3. Visualize Embeddings Sample ---
    print(f"\nStep 2: Visualizing Embeddings with Tags")
    try:
         if len(embeddings_df) > 0:
              # Subsample based on the (potentially reduced) aligned data
              sample_frac = 0.01
              num_samples_for_viz = max(200, int(len(embeddings_df) * sample_frac)) # Min 200 samples for viz
              num_samples_for_viz = min(num_samples_for_viz, len(embeddings_df)) # Cap at available data
              sample_indices = np.random.choice(embeddings_df.index, size=num_samples_for_viz, replace=False)

              embeddings_df_viz = embeddings_df.loc[sample_indices]
              processed_df_viz = processed_df.loc[sample_indices]
              print(f"Visualizing {len(embeddings_df_viz)} samples.")
              os.makedirs(embedding_tags_viz_dir, exist_ok=True)
              print(f"Generating plots for up to {len(tag_columns)} tags...")
              # Limit plotting if too many tags
              tags_to_plot = tag_columns[:min(len(tag_columns), 20)] if len(tag_columns) > 20 else tag_columns
              if len(tag_columns) > 20: print("  (Plotting first 20 tags only)")

              for tag_col in tqdm(tags_to_plot, desc="Generating Tag Plots"):
                  visualization.visualize_embeddings_with_tag(
                      embeddings_df=embeddings_df_viz, processed_df=processed_df_viz,
                      tag_column=tag_col, output_dir=embedding_tags_viz_dir,
                      show_clusters=True, n_clusters=5, random_state=random_state,
                      perplexity=30 # Use a fixed reasonable perplexity for samples
                  )
              print("Embedding visualization complete.")
         else:
              print("Warning: No data available for embedding visualization after alignment. Skipping.")
    except Exception as e: print(f"Error during embedding visualization: {e}"); traceback.print_exc()


    # --- 4. Prepare Data for K-Fold or Train/Test Split ---
    print(f"\nStep 3: Preparing Data for Classification Models...")
    tagged_df = processed_df[tag_columns].copy()
    # Ensure indices are still aligned before extracting values
    if not embeddings_df.index.equals(tagged_df.index):
         print("Error: Embeddings and Tags indices are unaligned before final modeling step.")
         # Attempt re-alignment one last time
         common_final_idx = embeddings_df.index.intersection(tagged_df.index)
         if len(common_final_idx) == 0: return None
         embeddings_df = embeddings_df.loc[common_final_idx]
         tagged_df = tagged_df.loc[common_final_idx]
         original_id_column = original_id_column.loc[common_final_idx]
         print("Re-aligned indices.")

    X = embeddings_df.values
    y = tagged_df.values
    # Use the final aligned original IDs/indices
    original_indices_mapping = original_id_column.values
    print(f"Data prepared for models: X shape: {X.shape}, y shape: {y.shape}, Original indices/IDs length: {len(original_indices_mapping)}")
    if X.shape[0] == 0: print("Error: No data remaining for model training."); return None

    # --- 5. K-Fold Cross-Validation or Single Split ---
    fold_metrics = defaultdict(lambda: defaultdict(list))
    all_models_trained = defaultdict(list) # Store models if needed (e.g., for later prediction)
    rf_feature_importances = [] # Store RF importances per fold
    final_metrics = {} # Store final aggregated metrics
    pca_model_used = None # Store the PCA model from the last fold/split if used

    if k_folds > 1:
        # K-Fold Cross-Validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        fold_num = 0
        for train_index, val_index in kf.split(X, y):
            fold_num += 1
            print(f"\n--- Fold {fold_num}/{k_folds} ---")
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            # Get original indices/IDs corresponding to the validation set for this fold
            val_original_indices = original_indices_mapping[val_index]

            # Apply scaling and optional PCA *within* the fold
            X_train_final, X_val_final, scaler, fold_pca = prepare_split_data(
                 X_train_fold, X_val_fold, reduce_dimensionality=fast_mode, random_state=random_state,
                 n_components=min(100, X_train_fold.shape[1] // 2) if fast_mode and X_train_fold.shape[1] > 1 else None,
                 pca_variance=0.95 # Target variance if n_components not set
            )
            if X_train_final is None: continue # Skip fold if preparation fails
            if fold_pca: pca_model_used = fold_pca # Keep track of PCA obj

            print(f"\nFold {fold_num}: Training Models...")
            rf_pred_fold = None # To store RF predictions for error analysis

            # --- Train Models for this Fold ---
            # Neural Network
            try:
                nn_model, nn_history, nn_metrics = modeling.train_neural_network(
                    X_train_final, X_val_final, y_train_fold, y_val_fold, tag_columns,
                    epochs=30 if not fast_mode else 10,
                    batch_size=64 if not fast_mode else 128
                )
                if nn_metrics:
                    for metric_name, value in nn_metrics.items(): fold_metrics['Neural Network'][metric_name].append(value)
                    if nn_history: visualization.plot_learning_curves(nn_history, "Neural_Network", fold_num, classification_viz_dir)
            except Exception as e: print(f"Error training NN in fold {fold_num}: {e}"); traceback.print_exc()

            # Tree-Based Models
            try:
                sample_size = min(5000, X_train_final.shape[0]) if fast_mode else None
                tree_results = modeling.train_tree_based_models(
                    X_train_final, X_val_final, y_train_fold, y_val_fold, tag_columns,
                    optimize_speed=fast_mode, sample_size=sample_size
                )
                rf_model, rf_metrics = tree_results.get('random_forest', (None, None))
                if rf_metrics:
                    for metric_name, value in rf_metrics.items(): fold_metrics['Random Forest'][metric_name].append(value)
                    # Store feature importances if available
                    if rf_model and isinstance(rf_model, MultiOutputClassifier) and hasattr(rf_model, 'estimators_') and rf_model.estimators_:
                        if hasattr(rf_model.estimators_[0], 'feature_importances_'):
                            fold_importances = np.mean([est.feature_importances_ for est in rf_model.estimators_], axis=0)
                            rf_feature_importances.append(fold_importances)
                        else: print("Warning: RF estimator lacks feature_importances_.")
                    # Get predictions for error analysis
                    if rf_model: rf_pred_fold = rf_model.predict(X_val_final)

                gb_model, gb_metrics = tree_results.get('gradient_boosting', (None, None))
                if gb_metrics:
                     for metric_name, value in gb_metrics.items(): fold_metrics['Gradient Boosting'][metric_name].append(value)
            except Exception as e: print(f"Error training Tree models in fold {fold_num}: {e}"); traceback.print_exc()

            # Logistic Regression
            try:
                lr_model, lr_metrics = modeling.train_regression_model(
                    X_train_final, X_val_final, y_train_fold, y_val_fold, tag_columns
                )
                if lr_metrics:
                    for metric_name, value in lr_metrics.items(): fold_metrics['Logistic Regression'][metric_name].append(value)
            except Exception as e: print(f"Error training LR in fold {fold_num}: {e}"); traceback.print_exc()

            # Error Analysis for this fold (using RF predictions)
            if rf_pred_fold is not None:
                print_error_analysis(y_val_fold, rf_pred_fold, val_original_indices, processed_df, tag_columns)

            print(f"--- Fold {fold_num} Completed ---")

        # Aggregate K-Fold Results
        print("\n--- Aggregating K-Fold Results ---")
        for model_name, metrics_dict in fold_metrics.items():
            if not metrics_dict: continue
            aggregated = {'model_type': model_name}
            print(f"\n{model_name}:")
            for metric_name, values_list in metrics_dict.items():
                if not values_list: continue
                # Handle per-tag dictionaries (like tag_f1_scores)
                if isinstance(values_list[0], dict):
                    avg_tag_metrics = defaultdict(float)
                    std_tag_metrics = defaultdict(list) # Store all values per tag
                    valid_fold_count = 0
                    for fold_dict in values_list:
                         if isinstance(fold_dict, dict):
                              for tag, score in fold_dict.items():
                                   avg_tag_metrics[tag] += score
                                   std_tag_metrics[tag].append(score)
                              valid_fold_count += 1
                         else:
                              print(f"Warning: Unexpected data type in per-tag metrics list for {model_name}, metric {metric_name}. Item: {type(fold_dict)}")
                    # Calculate average and std dev per tag
                    if valid_fold_count > 0:
                        final_avg_tags = {tag: total / valid_fold_count for tag, total in avg_tag_metrics.items()}
                        final_std_tags = {tag: np.std(scores) for tag, scores in std_tag_metrics.items()}
                    else:
                        final_avg_tags = {}
                        final_std_tags = {}
                    aggregated[metric_name] = final_avg_tags # Store the averaged dictionary
                    aggregated[f'{metric_name}_std'] = final_std_tags # Store std dev dictionary

                # Handle overall scalar metrics (like accuracy, f1_macro)
                elif isinstance(values_list[0], (int, float, np.number)):
                    numeric_values = [v for v in values_list if isinstance(v, (int, float, np.number)) and not np.isnan(v)]
                    if not numeric_values: continue
                    mean_val = np.mean(numeric_values); std_val = np.std(numeric_values)
                    aggregated[f'{metric_name}_mean'] = mean_val; aggregated[f'{metric_name}_std'] = std_val
                    print(f"  Avg {metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")
            final_metrics[model_name] = aggregated

        # Calculate average RF feature importances across folds
        avg_rf_importances = None
        if rf_feature_importances:
            avg_rf_importances = np.mean(rf_feature_importances, axis=0)
            print(f"\nCalculated average feature importances from {len(rf_feature_importances)} RF models.")

    else: # Single Train/Test Split
        print("Using single train/test split (80/20).")
        train_indices, test_indices = train_test_split(np.arange(len(X)), test_size=0.2, random_state=random_state, stratify=None) # Stratify might be complex for multilabel

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        test_original_indices = original_indices_mapping[test_indices]

        # Apply scaling and optional PCA to the split
        X_train, X_test, scaler, pca_model_used = prepare_split_data(
            X_train, X_test, reduce_dimensionality=fast_mode, random_state=random_state,
            n_components=min(100, X.shape[1] // 2) if fast_mode and X.shape[1] > 1 else None,
            pca_variance=0.95
        )
        if X_train is None: return None

        print(f"\nStep 4: Training and Evaluating Classification Models (Single Split)...")
        single_split_models = {} # Store trained models for this split
        rf_pred_single = None # For error analysis

        # --- Train Models for Single Split ---
        # Neural Network
        try:
            nn_model, nn_history, nn_metrics = modeling.train_neural_network(
                X_train, X_test, y_train, y_test, tag_columns,
                epochs=30 if not fast_mode else 10,
                batch_size=64 if not fast_mode else 128
            )
            if nn_metrics: final_metrics['Neural Network'] = nn_metrics
            if nn_model: single_split_models['neural_network'] = nn_model
            if nn_history: visualization.plot_learning_curves(nn_history, "Neural_Network", 0, classification_viz_dir)
        except Exception as e: print(f"Error training NN: {e}"); traceback.print_exc()

        # Tree-Based Models
        try:
            sample_size = min(5000, X_train.shape[0]) if fast_mode else None
            tree_results = modeling.train_tree_based_models(
                X_train, X_test, y_train, y_test, tag_columns,
                optimize_speed=fast_mode, sample_size=sample_size
            )
            rf_model, rf_metrics = tree_results.get('random_forest', (None, None))
            gb_model, gb_metrics = tree_results.get('gradient_boosting', (None, None))
            if rf_metrics: final_metrics['Random Forest'] = rf_metrics
            if rf_model:
                single_split_models['random_forest'] = rf_model
                rf_pred_single = rf_model.predict(X_test) # Get predictions for error analysis
            if gb_metrics: final_metrics['Gradient Boosting'] = gb_metrics
            if gb_model: single_split_models['gradient_boosting'] = gb_model
        except Exception as e: print(f"Error training Tree models: {e}"); traceback.print_exc()

        # Logistic Regression
        try:
            lr_model, lr_metrics = modeling.train_regression_model(
                X_train, X_test, y_train, y_test, tag_columns
            )
            if lr_metrics: final_metrics['Logistic Regression'] = lr_metrics
            if lr_model: single_split_models['logistic_regression'] = lr_model
        except Exception as e: print(f"Error training LR: {e}"); traceback.print_exc()

        # Error Analysis for single split (using RF predictions)
        if rf_pred_single is not None:
            print_error_analysis(y_test, rf_pred_single, test_original_indices, processed_df, tag_columns)

        # Add model_type key for consistency with CV results format
        for model_name in final_metrics.keys():
             if final_metrics[model_name] is not None:
                 final_metrics[model_name]['model_type'] = model_name

        # Feature importances are directly from the single trained RF model
        avg_rf_importances = None


    # --- 6. Report and Visualize Final Results ---
    print(f"\nStep 5: Reporting and Visualizing Final Model Results...")
    valid_final_metrics = {k: v for k, v in final_metrics.items() if v}
    if not valid_final_metrics:
        print("No valid model metrics to report or visualize.")
    else:
        print("\n===== Overall Performance Metrics =====")
        for model_name, metrics in valid_final_metrics.items():
            print(f"\n{model_name}:")
            if k_folds > 1:
                # Report CV means and std deviations
                acc_mean = metrics.get('accuracy_mean', np.nan)
                acc_std = metrics.get('accuracy_std', 0)
                hl_mean = metrics.get('hamming_loss_mean', np.nan)
                hl_std = metrics.get('hamming_loss_std', 0)
                f1_mean = metrics.get('f1_macro_mean', np.nan)
                f1_std = metrics.get('f1_macro_std', 0)
                print(f"  Accuracy (Exact Match): {acc_mean:.4f} (+/- {acc_std:.4f})" if not np.isnan(acc_mean) else "  Accuracy (Exact Match): N/A")
                print(f"  Hamming Loss: {hl_mean:.4f} (+/- {hl_std:.4f})" if not np.isnan(hl_mean) else "  Hamming Loss: N/A")
                print(f"  F1-Score (Macro): {f1_mean:.4f} (+/- {f1_std:.4f})" if not np.isnan(f1_mean) else "  F1-Score (Macro): N/A")
            else:
                # Report single split metrics
                acc = metrics.get('accuracy', np.nan)
                hl = metrics.get('hamming_loss', np.nan)
                f1 = metrics.get('f1_macro', np.nan)
                print(f"  Accuracy (Exact Match): {acc:.4f}" if not np.isnan(acc) else "  Accuracy (Exact Match): N/A")
                print(f"  Hamming Loss: {hl:.4f}" if not np.isnan(hl) else "  Hamming Loss: N/A")
                print(f"  F1-Score (Macro): {f1:.4f}" if not np.isnan(f1) else "  F1-Score (Macro): N/A")

        # Plot overall comparison
        visualization.plot_metrics_comparison(
             valid_final_metrics, output_dir=classification_viz_dir, use_cv_results=(k_folds > 1)
        )
        # Plot per-tag performance heatmap
        visualization.plot_tag_performance(
             valid_final_metrics, tag_columns, output_dir=classification_viz_dir
        )

        # Plot Feature Importance (using RF results)
        feature_names = None
        importance_source = None
        is_model_object_for_plot = False
        pca_applied = pca_model_used is not None

        if k_folds > 1:
            if avg_rf_importances is not None:
                print("\nAnalyzing feature importance (Averaged over K-Folds)...")
                importance_source = avg_rf_importances
                is_model_object_for_plot = False # It's an array
                num_features = importance_source.shape[0]
                if pca_applied: feature_names = [f'PC_{i+1}' for i in range(num_features)]
                else: feature_names = [f'Embed_{i+1}' for i in range(num_features)]
            else: print("\nRandom Forest feature importances not available from K-Fold run.")
        else: # Single split
            rf_model_single = single_split_models.get('random_forest')
            if rf_model_single:
                 print("\nAnalyzing feature importance (Single Split Model)...")
                 importance_source = rf_model_single
                 is_model_object_for_plot = True # It's a model object
                 try:
                     num_features = 0
                     if isinstance(rf_model_single, MultiOutputClassifier) and hasattr(rf_model_single, 'estimators_') and rf_model_single.estimators_:
                         if hasattr(rf_model_single.estimators_[0], 'n_features_in_'):
                             num_features = rf_model_single.estimators_[0].n_features_in_
                     elif hasattr(rf_model_single, 'n_features_in_'): # Should not happen for MultiOutput but as fallback
                          num_features = rf_model_single.n_features_in_

                     if num_features > 0:
                         if pca_applied: feature_names = [f'PC_{i+1}' for i in range(num_features)]
                         else: feature_names = [f'Embed_{i+1}' for i in range(num_features)]
                     else: print("Warning: Could not determine number of features from RF model.")
                 except Exception as feat_e: print(f"Warning: Error determining feature names: {feat_e}")

            else: print("\nRandom Forest model not available from single split run.")

        # Call plotting function if importance data is available
        if importance_source is not None:
            visualization.plot_feature_importance(
                 importances_data=importance_source,
                 is_model_object=is_model_object_for_plot,
                 feature_names=feature_names,
                 output_dir=classification_viz_dir, top_n=20
            )


    print("\nClassification Pipeline Finished.")
    # Return final metrics (either single split dict or aggregated CV dict)
    return final_metrics


# --- Autoencoder Training Pipeline ---
def run_autoencoder_pipeline(
    processed_data_path="data/processed/processed_recipes.csv",
    text_column='cleaned_text_for_lstm',
    ae_checkpoint_dir="checkpoints/lstm_autoencoder",
    ae_checkpoint_filename="lstm_autoencoder_checkpoint.pt",
    ae_vocab_filename="ae_vocab.pkl",
    max_len=100,
    embedding_dim=100,
    hidden_dim=128,
    batch_size=64,
    epochs=10,
    learning_rate=1e-3,
    run_preprocessing=False, # Option to run main preprocessing first
    raw_recipes_path="data/raw/RAW_recipes.csv",
    raw_interactions_path="data/raw/RAW_interactions.csv",
    ae_max_vocab=15000,
    ae_subsample_size=None, # Use None or 0 for full dataset
    random_state=42
    ):
    """Runs the LSTM Autoencoder training experiment, saves model and vocabulary."""
    print("Starting LSTM Autoencoder Training Pipeline...")

    # --- 1. Load or Generate Processed Data ---
    if run_preprocessing:
        print(f"\nStep 1a: Running Main Preprocessing first...")
        processed_df = preprocessing.preprocess_data(
            raw_recipes_path=raw_recipes_path,
            raw_interactions_path=raw_interactions_path,
            output_path=processed_data_path
        )
        if processed_df is None: return None
    else:
        print(f"\nStep 1a: Loading Preprocessed Data from {processed_data_path}...")
        try:
            processed_df = pd.read_csv(processed_data_path)
            processed_df = processed_df.reset_index(drop=True) # Ensure default index if loaded
            print(f"Loaded {len(processed_df)} processed recipes.")
        except FileNotFoundError:
            print(f"Error: Processed data file not found at {processed_data_path}.")
            return None
        except Exception as e: print(f"Error loading processed data: {e}"); return None

    if text_column not in processed_df.columns:
        print(f"Error: Text column '{text_column}' not found in loaded data.")
        return None

    # --- 2. Prepare Autoencoder Dataset ---
    print(f"\nStep 2: Preparing Autoencoder Dataset (Max Length: {max_len})...")
    texts_full = processed_df[text_column].astype(str).tolist()

    # Subsample if requested
    if ae_subsample_size is not None and ae_subsample_size > 0 and ae_subsample_size < len(texts_full):
        print(f"Using a subsample of {ae_subsample_size} texts for AE training.")
        random.seed(random_state)
        texts = random.sample(texts_full, ae_subsample_size)
    else:
        print(f"Using the full dataset ({len(texts_full)} texts) for AE training.")
        texts = texts_full

    # Create AE-specific dataset (handles vocab building, encoding, padding)
    print(f"Limiting AE vocabulary size to approximately {ae_max_vocab} tokens.")
    ae_dataset = lstm_autoencoder.RecipeTextDatasetAE(
        texts, vocab=None, max_len=max_len, max_vocab_size=ae_max_vocab
    )
    ae_dataloader = DataLoader(
        ae_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Save the built vocabulary
    os.makedirs(ae_checkpoint_dir, exist_ok=True)
    vocab_path = os.path.join(ae_checkpoint_dir, ae_vocab_filename)
    try:
        with open(vocab_path, 'wb') as f:
            pickle.dump(ae_dataset.vocab, f)
        print(f"Autoencoder vocabulary saved to {vocab_path}")
    except Exception as e:
        print(f"Error saving AE vocabulary: {e}")
        return None # Cannot proceed without vocab

    # --- 3. Initialize and Train LSTM Autoencoder ---
    print(f"\nStep 3: Initializing and Training LSTM Autoencoder...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get PAD index from the built vocabulary
    pad_idx = ae_dataset.vocab.get("<PAD>", 0)
    model = models.LSTMAutoencoder(
        vocab_size=len(ae_dataset.vocab),
        embedding_dim=embedding_dim, hidden_dim=hidden_dim, pad_idx=pad_idx
    )

    checkpoint_path = os.path.join(ae_checkpoint_dir, ae_checkpoint_filename)

    # Train the model using the dedicated function
    trained_model = lstm_autoencoder.train_autoencoder(
        model=model, dataloader=ae_dataloader, vocab_size=len(ae_dataset.vocab),
        checkpoint_path=checkpoint_path, device=device, epochs=epochs, learning_rate=learning_rate
    )

    print("\nLSTM Autoencoder Training Pipeline Finished.")


# --- Generate AE Embeddings Pipeline ---
def generate_ae_embeddings(
    processed_data_path="data/processed/processed_recipes.csv",
    text_column='cleaned_text_for_lstm',
    ae_checkpoint_dir="checkpoints/lstm_autoencoder",
    ae_checkpoint_filename="lstm_autoencoder_checkpoint.pt",
    ae_vocab_filename="ae_vocab.pkl",
    output_path="results/ae_full_text_embeddings.parquet",
    batch_size=128, # Inference batch size
    device=None
):
    """
    Loads a trained LSTM Autoencoder and its vocabulary, generates embeddings
    (using the encoder part) for the full dataset specified in processed_data_path,
    and saves them to a Parquet file with the original 'id' as index.
    """
    print("Starting Generate AE Embeddings Pipeline...")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Processed Data (Need 'id' and text column) ---
    print(f"\nStep 1: Loading Processed Data from {processed_data_path}...")
    try:
        processed_df = pd.read_csv(processed_data_path)
        if 'id' not in processed_df.columns:
             print("Error: 'id' column missing in processed data. Cannot generate embeddings with ID index.")
             return None
        original_id_column = processed_df['id'].copy()
        if not original_id_column.is_unique:
             print("Warning: 'id' column in processed data is not unique. Embeddings index might have duplicates.")
        # Keep original df structure for text access, use original_id_column for final index
        print(f"Loaded {len(processed_df)} processed recipes.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}.")
        return None
    except Exception as e: print(f"Error loading processed data: {e}"); return None

    if text_column not in processed_df.columns:
        print(f"Error: Text column '{text_column}' not found.")
        return None

    # --- 2. Load AE Vocabulary ---
    vocab_path = os.path.join(ae_checkpoint_dir, ae_vocab_filename)
    print(f"\nStep 2: Loading AE Vocabulary from {vocab_path}...")
    try:
        with open(vocab_path, 'rb') as f:
            ae_vocab = pickle.load(f)
        print(f"Loaded AE vocabulary with {len(ae_vocab)} tokens.")
    except FileNotFoundError:
        print(f"Error: AE Vocabulary file not found at {vocab_path}.")
        return None
    except Exception as e:
        print(f"Error loading AE vocabulary: {e}")
        return None

    # --- 3. Load Trained AE Model ---
    checkpoint_path = os.path.join(ae_checkpoint_dir, ae_checkpoint_filename)
    print(f"\nStep 3: Loading Trained AE Model from {checkpoint_path}...")
    try:
        # Load state dict first to get dimensions
        temp_state_dict = torch.load(checkpoint_path, map_location='cpu')
        embed_dim = temp_state_dict['embedding.weight'].shape[1]
        hidden_dim_key = next((k for k in temp_state_dict if 'encoder.weight_hh_l0' in k or 'encoder.hidden_size' in k), None)
        if hidden_dim_key:
            try:
                 hidden_dim = temp_state_dict['encoder.weight_hh_l0'].shape[1]
            except KeyError:
                 # Fallback if weight names differ
                 hidden_dim = temp_state_dict['decoder.weight_ih_l0'].shape[1]

        else:
            # Attempt to load a saved config if model saving included it, otherwise raise error
             raise KeyError("Could not automatically infer hidden_dim from state_dict keys. Save hidden_dim with model or adjust key names.")

        pad_idx = ae_vocab.get("<PAD>", 0)
        print(f"  Inferred AE model parameters: embed_dim={embed_dim}, hidden_dim={hidden_dim}")

        # Instantiate model and load state dict
        model = models.LSTMAutoencoder(
            vocab_size=len(ae_vocab), embedding_dim=embed_dim, hidden_dim=hidden_dim, pad_idx=pad_idx
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("AE Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: AE Model checkpoint not found at {checkpoint_path}.")
        return None
    except Exception as e:
        print(f"Error loading AE model state: {e}")
        traceback.print_exc()
        return None

    # --- 4. Prepare Dataset for AE Embedding Generation ---
    print("\nStep 4: Preparing Dataset for AE Embedding Generation...")
    max_len = 100
    # Use the AE Dataset class again for consistent cleaning, encoding, padding
    ae_inference_dataset = lstm_autoencoder.RecipeTextDatasetAE(
        processed_df[text_column].astype(str).tolist(),
        vocab=ae_vocab, max_len=max_len, max_vocab_size=None # Don't limit vocab here
    )
    # Use SequentialSampler for inference to maintain order
    inference_sampler = SequentialSampler(ae_inference_dataset)
    ae_dataloader = DataLoader(
        ae_inference_dataset, batch_size=batch_size, sampler=inference_sampler, num_workers=0
    )

    # --- 5. Generate Embeddings using AE Encoder ---
    print("\nStep 5: Generating Embeddings using AE Encoder...")
    all_embeddings_list = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for inputs, _ in tqdm(ae_dataloader, desc="Generating AE embeddings"):
            inputs = inputs.to(device)
            # Use the model's encode method
            embedding_batch = model.encode(inputs)
            # Move embeddings to CPU and convert to numpy
            all_embeddings_list.append(embedding_batch.cpu().numpy())

    if not all_embeddings_list:
        print("Error: No embeddings were generated.")
        return None

    # --- 6. Create DataFrame and Save ---
    all_embeddings_np = np.vstack(all_embeddings_list)
    # Use the original 'id' column values as the index for the final DataFrame
    all_embeddings_df = pd.DataFrame(
        all_embeddings_np, index=original_id_column, columns=[f"ae_embed_{i}" for i in range(all_embeddings_np.shape[1])]
    )
    all_embeddings_df.index.name = 'id' # Name the index column

    print(f"\nAE Embeddings generated for all {len(all_embeddings_df)} samples.")
    print(f"AE Embeddings dimension: {all_embeddings_np.shape[1]}")

    print(f"\nStep 7: Saving AE Embeddings to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_embeddings_df.to_parquet(output_path)
        print(f"Successfully saved AE embeddings to {output_path}")
    except Exception as e:
        print(f"Error saving AE embeddings: {e}")

    print("\nGenerate AE Embeddings Pipeline Finished.")
    return all_embeddings_df


# --- Generate SBERT Embeddings Pipeline ---
def generate_sbert_embeddings(
    processed_data_path="data/processed/processed_recipes.csv",
    text_column='cleaned_text_for_lstm',
    model_name='all-MiniLM-L6-v2',
    output_path="results/sbert_embeddings.parquet",
    batch_size=64,
    device=None
):
    """
    Generates Sentence-BERT embeddings for the specified text column and
    saves them to a Parquet file with the original 'id' as index.
    """
    print("Starting Generate SBERT Embeddings Pipeline...")

    if SentenceTransformer is None:
        print("Error: 'sentence-transformers' library not installed. Cannot generate SBERT embeddings.")
        print("Please install it: pip install sentence-transformers")
        return None

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Processed Data ---
    print(f"\nStep 1: Loading Processed Data from {processed_data_path}...")
    try:
        processed_df = pd.read_csv(processed_data_path)
        if 'id' not in processed_df.columns:
             print("Error: 'id' column missing in processed data. Cannot generate embeddings with ID index.")
             return None
        if not processed_df['id'].is_unique:
             print("Warning: 'id' column in processed data is not unique. Index may have duplicates.")
        # Store the original 'id' column to use as index later
        original_id_column = processed_df['id'].copy()
        print(f"Loaded {len(processed_df)} processed recipes.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}.")
        return None
    except Exception as e: print(f"Error loading processed data: {e}"); return None

    if text_column not in processed_df.columns:
        print(f"Error: Text column '{text_column}' not found.")
        return None

    texts_to_encode = processed_df[text_column].astype(str).tolist()

    # --- 2. Load SBERT Model ---
    print(f"\nStep 2: Loading SBERT model '{model_name}'...")
    try:
        sbert_model = SentenceTransformer(model_name, device=device)
        print(f"SBERT model loaded. Embedding dimension: {sbert_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Error loading SBERT model '{model_name}': {e}")
        traceback.print_exc()
        return None

    # --- 3. Generate Embeddings ---
    print("\nStep 3: Generating SBERT Embeddings...")
    try:
        sbert_embeddings_np = sbert_model.encode(
            texts_to_encode,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True # Efficiently get numpy array
        )
    except Exception as e:
        print(f"Error during SBERT encoding: {e}")
        traceback.print_exc()
        return None

    if sbert_embeddings_np is None or len(sbert_embeddings_np) != len(processed_df):
        print("Error: SBERT embedding generation failed or returned incorrect number of embeddings.")
        return None

    # --- 4. Create and Save DataFrame ---
    sbert_embeddings_df = pd.DataFrame(
        sbert_embeddings_np,
        index=original_id_column, # Use the original 'id' column values as index
        columns=[f"sbert_embed_{i}" for i in range(sbert_embeddings_np.shape[1])]
    )
    sbert_embeddings_df.index.name = 'id' # Name the index 'id'

    print(f"\nSBERT Embeddings generated for all {len(sbert_embeddings_df)} samples.")
    print(f"SBERT Embeddings dimension: {sbert_embeddings_np.shape[1]}")

    print(f"\nStep 5: Saving SBERT Embeddings to {output_path}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sbert_embeddings_df.to_parquet(output_path)
        print(f"Successfully saved SBERT embeddings to {output_path}")
    except Exception as e:
        print(f"Error saving SBERT embeddings: {e}")

    print("\nGenerate SBERT Embeddings Pipeline Finished.")
    return sbert_embeddings_df


# Helper Function to prepare data within a split/fold
def prepare_split_data(X_train, X_test, reduce_dimensionality=True, random_state=42,
                       n_components=None, pca_variance=0.95):
    """
    Applies StandardScaler and optional PCA to train/test splits.
    PCA is fitted only on the training data and applied to both.
    """
    print(f"\n--- Applying Scaling {'and PCA ' if reduce_dimensionality else ''}to Data Split ---")
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Data scaled. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    pca = None
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled

    # Optional PCA
    if reduce_dimensionality:
        print(f"PCA requested. Original dims: {X_train_scaled.shape[1]}", end="")
        max_possible_comp = min(X_train_scaled.shape) # Max components is min(n_samples, n_features)

        if n_components is None:
            # PCA based on variance explained
            pca_variance = max(0.01, min(0.99, pca_variance))
            # Ensure n_components for variance doesn't exceed max possible
            pca_components_for_variance = min(max_possible_comp, X_train_scaled.shape[1])
            pca = PCA(n_components=pca_components_for_variance, random_state=random_state)
            print(f", Target variance: {pca_variance:.2f} (using up to {pca_components_for_variance} components)")
        else:
            # PCA based on fixed number of components
            # Ensure n_components is valid
            n_components = max(1, min(n_components, max_possible_comp)) if max_possible_comp > 0 else 0
            pca = PCA(n_components=n_components, random_state=random_state)
            print(f", Target components: {n_components}")

        if pca is not None and pca.n_components > 0:
             try:
                  X_train_pca_temp = pca.fit_transform(X_train_scaled)
                  # If PCA was based on variance, select components that reach the target
                  if n_components is None:
                       cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                       n_comps_for_variance = np.argmax(cumulative_variance >= pca_variance) + 1
                       print(f"\n  -> Selected {n_comps_for_variance} components to explain {pca_variance:.2f} variance (Actual: {cumulative_variance[n_comps_for_variance-1]:.4f})")
                       # Re-fit PCA with the determined number of components
                       pca = PCA(n_components=n_comps_for_variance, random_state=random_state)
                       X_train_final = pca.fit_transform(X_train_scaled)
                  else:
                       X_train_final = X_train_pca_temp # Already fitted with fixed n_components

                  X_test_final = pca.transform(X_test_scaled)
                  print(f"PCA applied. Reduced dims: {X_train_final.shape[1]}")

             except ValueError as e:
                  print(f"\nError during PCA: {e}. Skipping PCA.")
                  X_train_final = X_train_scaled
                  X_test_final = X_test_scaled
                  pca = None # Ensure pca object is None if it failed
        else:
             print("\nSkipping PCA: Invalid number of components requested or computed.")
             pca = None
             X_train_final = X_train_scaled
             X_test_final = X_test_scaled
    else:
        print("PCA not applied.")

    print(f"Final Train shape: {X_train_final.shape}, Final Test shape: {X_test_final.shape}")
    # Return final data, the scaler, and the fitted PCA object (or None)
    return X_train_final, X_test_final, scaler, pca