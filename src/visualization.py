# NewDesign/src/visualization.py

import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import warnings
import traceback

# Set plot styling defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


def visualize_embeddings_with_tag(
    embeddings_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    tag_column: str,
    output_dir: str = "results/visualizations/embedding_tags_lstm",
    show_clusters: bool = True,
    n_clusters: int = 5,
    random_state: int = 42,
    perplexity: int = 30,
    use_subsample: float = None
):
    """
    Visualize embeddings using t-SNE with coloring based on a specific tag column.
    Optionally includes K-Means clustering visualization.
    Handles index alignment between embeddings and processed data.
    """
    print(f"\n--- Visualizing Embeddings for Tag: {tag_column} ---")
    base_filename = f"embeddings_{tag_column}.png"
    filepath = os.path.join(output_dir, base_filename)
    combined_base_filename = f"embeddings_{tag_column}_combined.png"
    combined_filepath = os.path.join(output_dir, combined_base_filename)

    os.makedirs(output_dir, exist_ok=True)

    try:
        # --- Data Preparation & Alignment ---
        common_index = embeddings_df.index.intersection(processed_df.index)
        if len(common_index) == 0:
             print("Error: Embeddings and Processed dataframes have no common indices. Cannot visualize.")
             return None
        if len(common_index) < len(embeddings_df) or len(common_index) < len(processed_df):
            print(f"Warning: Aligning embeddings ({len(embeddings_df)}) and processed data ({len(processed_df)}) on common indices ({len(common_index)}).")

        embeddings_subset = embeddings_df.loc[common_index]
        processed_subset = processed_df.loc[common_index]

        if use_subsample is not None and 0 < use_subsample < 1:
            print(f"Sampling {use_subsample*100:.1f}% of aligned data for visualization...")
            sample_size = max(1, int(len(embeddings_subset) * use_subsample))
            sampled_indices = np.random.choice(embeddings_subset.index, size=sample_size, replace=False)
            embeddings_subset = embeddings_subset.loc[sampled_indices]
            processed_subset = processed_subset.loc[sampled_indices]
            print(f"Using {len(embeddings_subset)} samples.")

        embeddings = embeddings_subset.values
        current_perplexity = min(perplexity, len(embeddings) - 1)
        if current_perplexity <= 0:
             print(f"Warning: Too few samples ({len(embeddings)}) for t-SNE perplexity={perplexity}. Skipping plot.")
             return None

        # --- t-SNE ---
        print(f"Applying t-SNE dimensionality reduction (perplexity={current_perplexity})...")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=current_perplexity, n_jobs=-1, init='pca', learning_rate='auto')
        reduced_embeddings = tsne.fit_transform(embeddings)

        plot_df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1]
        }, index=embeddings_subset.index) # Keep original index for joins

        # --- K-Means Clustering ---
        n_clusters_adjusted = n_clusters
        if show_clusters:
            print(f"Applying K-Means clustering (k={n_clusters})...")
            n_clusters_adjusted = min(n_clusters, len(embeddings))
            if n_clusters_adjusted < 2:
                 print(f"Warning: Too few samples ({len(embeddings)}) for K-Means k={n_clusters}. Skipping clustering visualization.")
                 plot_df['cluster'] = 0
                 show_clusters = False # Disable cluster plotting if k < 2
            else:
                 kmeans = KMeans(n_clusters=n_clusters_adjusted, random_state=random_state, n_init=10)
                 clusters = kmeans.fit_predict(embeddings)
                 plot_df['cluster'] = clusters

        # --- Add Tag Data ---
        if tag_column not in processed_subset.columns:
             print(f"Error: Tag column '{tag_column}' not found in processed data.")
             plot_df['tag'] = -1 # Assign placeholder value
        else:
             # Direct assignment since processed_subset is aligned with plot_df index
             plot_df['tag'] = processed_subset[tag_column]

        # --- Plotting: Separate Cluster and Tag Plots ---
        plt.figure(figsize=(18 if show_clusters else 10, 10))

        if show_clusters:
            # Plot Clusters
            ax1 = plt.subplot(1, 2, 1)
            sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', palette='viridis',
                            alpha=0.6, s=20, legend='full', ax=ax1)
            ax1.set_title(f't-SNE by Cluster (k={n_clusters_adjusted})')
            ax1.set_xlabel("t-SNE Component 1")
            ax1.set_ylabel("t-SNE Component 2")
            # Plot Tags in second subplot
            ax2 = plt.subplot(1, 2, 2)
        else:
            # Plot only Tags
            ax2 = plt.gca()

        # Plot Tags (only points with 0 or 1)
        valid_tags_df = plot_df[plot_df['tag'].isin([0, 1])]
        if not valid_tags_df.empty:
            sns.scatterplot(data=valid_tags_df, x='x', y='y', hue='tag',
                            palette={0: 'blue', 1: 'red'}, # Explicit color mapping
                            alpha=0.6, s=20, ax=ax2)
            ax2.set_title(f't-SNE by Tag: {tag_column}')
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")
            # Customize legend for clarity
            handles, labels = ax2.get_legend_handles_labels()
            new_labels = [f"No" if label == '0' else f"Yes" if label == '1' else label for label in labels]
            legend_title = tag_column.replace('tag_', '').replace('_', ' ').title()
            ax2.legend(handles=handles, labels=new_labels, title=legend_title)
        else:
            # Handle case where no valid tags exist in the sample
            ax2.text(0.5, 0.5, "No data points with valid tags (0 or 1) found.",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f't-SNE by Tag: {tag_column} (No Valid Data)')

        plt.suptitle(f"t-SNE Visualization for {tag_column}", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(filepath)
        plt.close()
        print(f"Saved tag visualization to '{filepath}'")

        # --- Plotting: Combined Plot (Color=Tag, Shape=Cluster) ---
        if show_clusters and not valid_tags_df.empty:
             plt.figure(figsize=(12, 10))
             markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'H'] # More markers if needed

             sns.scatterplot(
                 data=valid_tags_df,
                 x='x',
                 y='y',
                 hue='tag',        # Color by tag presence
                 style='cluster',  # Shape by cluster assignment
                 palette={0: 'blue', 1: 'red'},
                 markers=markers[:n_clusters_adjusted], # Use appropriate number of markers
                 alpha=0.7,
                 s=35 # Slightly larger points
             )

             plt.title(f't-SNE: Clusters (Shape) vs Tag "{tag_column}" (Color)')
             plt.xlabel("t-SNE Component 1")
             plt.ylabel("t-SNE Component 2")
             # Place legend outside plot area
             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
             plt.savefig(combined_filepath)
             plt.close()
             print(f"Saved combined visualization to '{combined_filepath}'")

        return plot_df

    except Exception as e:
        print(f"Error during visualization for tag '{tag_column}': {e}")
        traceback.print_exc()
        return None

# --- Model Evaluation Plotting Functions ---

def plot_metrics_comparison(metrics_dict, output_dir="results/visualizations/classification", use_cv_results=False):
    """
    Plots comparison of Accuracy, F1-Score, and Hamming Loss across models.
    Handles both single run metrics and averaged CV metrics with std dev.
    """
    print("\nPlotting model performance comparison...")
    os.makedirs(output_dir, exist_ok=True)

    models = list(metrics_dict.keys())
    if not models:
        print("No model metrics found to plot.")
        return

    # Extract metric values, checking for appropriate keys based on CV status
    if use_cv_results:
        accuracies_mean = [metrics_dict[model].get('accuracy_mean', np.nan) for model in models]
        f1_scores_mean = [metrics_dict[model].get('f1_macro_mean', np.nan) for model in models]
        hamming_losses_mean = [metrics_dict[model].get('hamming_loss_mean', np.nan) for model in models]
        accuracies_std = [metrics_dict[model].get('accuracy_std', 0) for model in models]
        f1_scores_std = [metrics_dict[model].get('f1_macro_std', 0) for model in models]
        hamming_losses_std = [metrics_dict[model].get('hamming_loss_std', 0) for model in models]
    else:
        # Use non-'_mean' keys for single split
        accuracies_mean = [metrics_dict[model].get('accuracy', np.nan) for model in models]
        f1_scores_mean = [metrics_dict[model].get('f1_macro', np.nan) for model in models]
        hamming_losses_mean = [metrics_dict[model].get('hamming_loss', np.nan) for model in models]
        # No standard deviation for single split
        accuracies_std = None
        f1_scores_std = None
        hamming_losses_std = None

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    # Internal helper function for plotting bars
    def plot_bar(axis, data_mean, data_std, title, color):
        indices = np.arange(len(models))
        # Only pass yerr if data_std is not None (i.e., use_cv_results is True)
        error_kw_settings = {'elinewidth':1, 'capthick':1} if data_std is not None else None
        bars = axis.bar(indices, data_mean, color=color, yerr=data_std, capsize=5, alpha=0.8,
                        error_kw=error_kw_settings)
        axis.set_title(title)
        axis.set_ylabel(title.split('(')[0].strip())
        axis.set_xticks(indices)
        axis.set_xticklabels(models, rotation=15, ha="right")
        valid_means = [m for m in data_mean if not np.isnan(m)]
        if valid_means:
            # Adjust ylim dynamically, ensuring it's at least 0-1
            upper_limit = max(1.0, max(valid_means) * 1.15)
            axis.set_ylim(0, upper_limit)
        else:
            axis.set_ylim(0, 1.0)
        # Add text labels above bars
        for i, (bar, mean_val) in enumerate(zip(bars, data_mean)):
             if not np.isnan(mean_val):
                 # Adjust text position based on ylim
                 y_pos = mean_val + (axis.get_ylim()[1] * 0.02)
                 axis.text(bar.get_x() + bar.get_width() / 2, y_pos,
                           f"{mean_val:.3f}", ha='center', va='bottom', fontsize=9)

    plot_bar(ax[0], accuracies_mean, accuracies_std, 'Accuracy (Exact Match Ratio)', 'skyblue')
    plot_bar(ax[1], f1_scores_mean, f1_scores_std, 'F1-Score (Macro Average)', 'lightgreen')
    plot_bar(ax[2], hamming_losses_mean, hamming_losses_std, 'Hamming Loss (Lower is Better)', 'salmon')

    plot_title = 'Multi-Label Classification Model Performance Comparison'
    if use_cv_results:
        plot_title += ' (Avg over K-Folds)'
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    filename = "model_comparison_metrics.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved model comparison plot to '{filepath}'")


def plot_tag_performance(metrics_dict, tag_columns, output_dir="results/visualizations/classification"):
    """Plots heatmaps of per-tag F1-score and Accuracy across different models."""
    print("\nPlotting per-tag performance...")
    os.makedirs(output_dir, exist_ok=True)

    models = list(metrics_dict.keys())
    valid_models = [m for m in models if metrics_dict.get(m) is not None]
    if not valid_models:
        print("No valid model metrics found to plot tag performance.")
        return

    for metric in ['f1_score', 'accuracy']:
        tag_perf = {}
        # Adjust key based on metric name convention used in pipelines.py
        metric_key = 'tag_f1_scores' if metric == 'f1_score' else 'tag_accuracies'

        for model_name in valid_models:
            per_tag_data = metrics_dict[model_name].get(metric_key)
            if isinstance(per_tag_data, dict):
                # Ensure data aligns with tag_columns order
                tag_perf[model_name] = [per_tag_data.get(tag, np.nan) for tag in tag_columns]
            else:
                print(f"Warning: Metric key '{metric_key}' not found or not a dict for model '{model_name}'. Skipping.")
                tag_perf[model_name] = [np.nan] * len(tag_columns) # Fill with NaN

        if not tag_perf:
             print(f"No performance data found for metric '{metric}'. Skipping plot.")
             continue

        tag_perf_df = pd.DataFrame(tag_perf, index=tag_columns)

        plt.figure(figsize=(10, max(8, len(tag_columns) // 4)), dpi=100)
        sns.heatmap(tag_perf_df, annot=True, fmt=".3f", cmap='YlGnBu', vmin=0, vmax=1, linewidths=.5, cbar=True, square=False, annot_kws={"size": 8})
        plot_title = 'Per-Tag F1 Scores by Model' if metric == 'f1_score' else 'Per-Tag Accuracies by Model'
        plt.title(plot_title)
        plt.ylabel('Tag')
        plt.xlabel('Model')
        plt.xticks(rotation=15, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        filename = f"tag_perf_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved tag {metric} comparison plot to '{filepath}'")

        # Analyze best/worst tags
        try:
            # Calculate average performance per tag across models
            tag_perf_df['average'] = tag_perf_df.mean(axis=1, skipna=True)
            tag_perf_df.dropna(subset=['average'], inplace=True) # Remove tags with no valid scores

            if not tag_perf_df.empty:
                best_tags = tag_perf_df.nlargest(5, 'average')
                worst_tags = tag_perf_df.nsmallest(5, 'average')

                print(f"\nTop 5 best performing tags ({metric} average across models):")
                print(best_tags[['average']].to_string(float_format="%.4f"))

                print(f"\nTop 5 worst performing tags ({metric} average across models):")
                print(worst_tags[['average']].to_string(float_format="%.4f"))
            else:
                print(f"No valid average tag performance calculated for metric '{metric}'.")
        except Exception as e:
             print(f"Could not calculate best/worst tags: {e}")


def plot_feature_importance(importances_data, is_model_object, feature_names=None, output_dir="results/visualizations/classification", top_n=20):
    """
    Plots feature importances based on a model object or a pre-computed array.
    Handles MultiOutputClassifier by averaging importances.
    """
    os.makedirs(output_dir, exist_ok=True)
    importances_array = None
    plot_title_suffix = ""

    try:
        if is_model_object:
            model = importances_data
            # Handle MultiOutputClassifier (like RF, GB)
            if isinstance(model, MultiOutputClassifier) and hasattr(model, 'estimators_') and model.estimators_:
                 if hasattr(model.estimators_[0], 'feature_importances_'):
                     # Average importances across all estimators (one per tag)
                     importances_array = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                     plot_title_suffix = " (Averaged over Tags)"
                     print("Averaging feature importances across model estimators (tags).")
                 else:
                      print("Warning: MultiOutputClassifier estimators lack 'feature_importances_'. Cannot plot.")
                      return
            # Handle single estimator models (less common here, but possible)
            elif hasattr(model, 'feature_importances_'):
                 importances_array = model.feature_importances_
                 print("Using feature importances from the provided model object directly.")
            else:
                 print(f"Warning: Provided model object type '{type(model)}' is not supported or not trained for feature importance.")
                 return
        else:
            # Assume pre-computed array (e.g., averaged from K-Fold)
            importances_array = importances_data
            if not isinstance(importances_array, np.ndarray) or importances_array.ndim != 1:
                 print(f"Warning: Expected 1D numpy array for pre-computed importances, received type {type(importances_array)}.")
                 return
            plot_title_suffix = " (Averaged over K-Folds)"
            print("Using pre-computed feature importances array.")

        if importances_array is None or len(importances_array) == 0:
            print("No feature importances available to plot.")
            return

        num_features = len(importances_array)
        top_n = min(top_n, num_features)
        if top_n <= 0:
            print("No features to plot (top_n <= 0).")
            return

        # Get indices of top N features
        indices = np.argsort(importances_array)[-top_n:]

        # Determine feature labels
        if feature_names is None:
             # Default labels if names not provided
             labels = [f'Feature {i}' for i in indices]
        elif len(feature_names) == num_features:
             # Use provided names
             labels = [feature_names[i] for i in indices]
        else:
             # Fallback if length mismatch
             print(f"Warning: Length mismatch between feature_names ({len(feature_names)}) and importances ({num_features}). Using generic labels.")
             labels = [f'Feature {i}' for i in indices]

        plt.figure(figsize=(10, max(6, top_n * 0.4)), dpi=100)
        plt.barh(range(len(indices)), importances_array[indices], align='center', color='lightcoral')
        plt.yticks(range(len(indices)), labels)
        plt.xlabel('Feature Importance' + plot_title_suffix)
        plt.title(f'Top {top_n} Important Features')
        plt.gca().invert_yaxis() # Display most important at the top
        plt.tight_layout()

        filename = 'feature_importance.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved feature importance plot to '{filepath}'")

        # Print top features to console
        print(f"\nTop {top_n} important features{plot_title_suffix}:")
        # Sort labels and scores descending for printing
        top_labels_sorted = labels[::-1]
        top_importances_sorted = importances_array[indices][::-1]
        for i in range(top_n):
             name = top_labels_sorted[i]
             score = top_importances_sorted[i]
             print(f"{i+1}. {name}: {score:.4f}")

    except Exception as e:
        print(f"Error during feature importance analysis/plotting: {e}")
        traceback.print_exc()


def plot_learning_curves(history, model_name, fold_num, output_dir="results/visualizations/classification"):
    """
    Plots training & validation loss and accuracy curves from Keras history.
    Handles potential variations in accuracy key names.
    """
    if not hasattr(history, 'history') or not isinstance(history.history, dict):
         print(f"Warning: Invalid or empty history object provided for {model_name}. Skipping learning curve plot.")
         return

    print(f"Plotting learning curves for {model_name}" + (f" (Fold {fold_num})" if fold_num > 0 else " (Single Split)"))
    os.makedirs(output_dir, exist_ok=True)

    history_dict = history.history
    loss = history_dict.get('loss')
    val_loss = history_dict.get('val_loss')

    # Find accuracy keys robustly (e.g., 'accuracy', 'acc', 'binary_accuracy')
    acc_key = next((k for k in history_dict if 'accuracy' in k.lower() and 'val' not in k.lower()), None)
    val_acc_key = next((k for k in history_dict if 'accuracy' in k.lower() and 'val' in k.lower()), None)

    if not loss or not val_loss or not acc_key or not val_acc_key:
        print(f"Warning: History object for {model_name} missing required keys (loss, val_loss, accuracy, val_accuracy). Keys found: {list(history_dict.keys())}. Cannot plot.")
        return

    acc = history_dict[acc_key]
    val_acc = history_dict[val_acc_key]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo-', label=f'Training {acc_key}')
    plt.plot(epochs, val_acc, 'ro-', label=f'Validation {val_acc_key}')
    plt.title(f'{model_name} - Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fold_suffix = f"_fold_{fold_num}" if fold_num > 0 else ""
    filename = f"learning_curves_{model_name.lower().replace(' ', '_')}{fold_suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved learning curves plot to '{filepath}'")


def plot_ablation_summary(ablation_results, output_dir="results/visualizations", metric_key='f1_macro', model_name='Random Forest'):
    """
    Plots a summary comparison of a specified metric across ablation study configurations
    for a specified model. Handles CV and single-split results.
    """
    print(f"\nPlotting ablation study summary for '{model_name}' - Metric: '{metric_key}'...")
    os.makedirs(output_dir, exist_ok=True)

    # Expects keys to be the config identifiers (e.g., N_TAGS values)
    config_names = list(ablation_results.keys())
    if not config_names:
         print("No ablation results provided.")
         return
    # Sort config names numerically if possible for better plotting order
    try:
        config_names = sorted(config_names, key=int)
    except ValueError:
        config_names = sorted(config_names) # Sort alphabetically if not numbers


    metric_values = []
    error_bars = []
    is_cv_result = False # Flag to track if any config used CV

    # Define keys for CV and single split results
    cv_metric_key = f"{metric_key}_mean"
    single_metric_key = metric_key
    std_dev_key = f"{metric_key}_std"

    results_found = False

    for name in config_names:
        # Results for a given config name (e.g., n_tags=5)
        config_result_dict = ablation_results.get(name, {})
        # Results for the specific model within that config (e.g., Random Forest)
        target_model_data = config_result_dict.get(model_name, {})

        # Check for CV mean first, fallback to single split key
        mean_val = target_model_data.get(cv_metric_key, target_model_data.get(single_metric_key, np.nan))
        # Get standard deviation if present (only for CV)
        std_val = target_model_data.get(std_dev_key, 0)

        metric_values.append(mean_val)
        error_bars.append(std_val)

        # Mark if any valid result was found and if any used CV
        if not np.isnan(mean_val):
            results_found = True
            if (std_dev_key in target_model_data and target_model_data[std_dev_key] > 0) or cv_metric_key in target_model_data:
                 is_cv_result = True


    if not results_found:
         print(f"No valid results found for model '{model_name}' and metric '{metric_key}' to plot for ablation study.")
         return

    # --- Plotting ---
    plt.figure(figsize=(max(10, len(config_names) * 1.5), 6))
    indices = np.arange(len(config_names))
    # Only plot error bars if it's CV data and there's actual std deviation > 0
    plot_error_bars = is_cv_result and any(e > 0 for e in error_bars if not np.isnan(e))

    bar_container = plt.bar(indices, metric_values, yerr=error_bars if plot_error_bars else None,
                            capsize=5, color='teal', alpha=0.8,
                            error_kw={'elinewidth':1, 'capthick':1} if plot_error_bars else None)

    ylabel = metric_key.replace('_', ' ').title()
    if is_cv_result: ylabel = f"Average {ylabel}"
    ylabel += f" - {model_name}"
    plt.ylabel(ylabel)

    # Use config names (convert to string for display) on x-axis
    plt.xticks(indices, [str(name) for name in config_names])
    plt.xlabel("Configuration (e.g., Number of Top Tags)")
    plt.title(f'Ablation Study: {model_name} Performance Comparison')

    # Set y-limits dynamically based on data range
    valid_values = [v for v in metric_values if not np.isnan(v)]
    if valid_values:
         min_val = min(valid_values) if min(valid_values) > 0 else 0
         max_val = max(valid_values)
         padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
         # Sensible upper limit, potentially capped at 1.05 for scores like F1/Accuracy
         upper_lim = 1.05 if ('accuracy' in metric_key or 'f1' in metric_key) and max_val <= 1 else max_val + padding*1.5
         upper_lim = max(upper_lim, max_val + 0.01) # Ensure slightly above max if max is low
         # Ensure lower limit is slightly below min value unless min is 0
         lower_lim = min_val - padding if min_val > padding else 0
         plt.ylim(lower_lim, upper_lim)
    else:
         plt.ylim(0, 1.05) # Default if no valid data

    plt.grid(axis='y', linestyle='--')

    # Add text labels to bars
    plt.bar_label(bar_container, fmt='%.3f', padding=3)

    plt.tight_layout()
    filename = f"ablation_summary_{model_name.lower().replace(' ', '_')}_{metric_key}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved ablation summary plot to '{filepath}'")


# --- Analysis Functions ---

def find_similar_texts(query_identifier, is_positional, embeddings_df, text_df, text_column, top_n=5):
    """
    Find texts similar to a query text based on embedding cosine similarity.
    Handles both positional index and ID-based index lookup for the query.

    Args:
        query_identifier (int): The positional index or recipe ID to query.
        is_positional (bool): True if query_identifier is a positional index (0-based),
                             False if it's a recipe ID to be looked up in the index.
        embeddings_df (pd.DataFrame): DataFrame containing embeddings. Index can be RangeIndex or 'id'.
        text_df (pd.DataFrame): DataFrame containing original texts. Should have 'id' column
                                and ideally 'id' as index for robust lookup.
        text_column (str): Column name in text_df containing the text.
        top_n (int): Number of similar texts to return.
    """
    print(f"\nFinding texts similar to {'positional index' if is_positional else 'recipe ID'}: {query_identifier}...")

    query_embedding = None
    query_text = "Query text not found"
    query_actual_index = None # The index label (ID or position) used for the embedding

    # --- Locate Query Embedding and Text ---
    try:
        if is_positional:
            if query_identifier < 0 or query_identifier >= len(embeddings_df):
                print(f"Error: Positional index {query_identifier} is out of bounds for embeddings DataFrame (len={len(embeddings_df)}).")
                return None
            # Get the index label (could be int position or an ID) at the specified position
            query_actual_index = embeddings_df.index[query_identifier]
            # Retrieve embedding using iloc for position-based access
            query_embedding = embeddings_df.iloc[[query_identifier]].values
        else: # Query identifier is an ID
            if query_identifier not in embeddings_df.index:
                print(f"Error: Recipe ID {query_identifier} not found in embeddings DataFrame index.")
                return None
            query_actual_index = query_identifier
            # Retrieve embedding using loc for label-based access
            query_embedding = embeddings_df.loc[[query_identifier]].values

        # Attempt to get query text using the located index label (query_actual_index)
        # Prioritize lookup using text_df's index if it matches the query index type/value
        if query_actual_index in text_df.index:
             query_text = text_df.loc[query_actual_index, text_column]
        # Fallback: Check if 'id' column exists and contains the index label
        elif 'id' in text_df.columns and query_actual_index in text_df['id'].values:
             # Find first row matching the ID and get text (less robust if IDs aren't unique)
             query_text = text_df[text_df['id'] == query_actual_index][text_column].iloc[0]
        else:
             print(f"Warning: Could not reliably locate query text for index/ID {query_actual_index} in text_df.")

    except Exception as e:
        print(f"Error locating query embedding or text for identifier {query_identifier}: {e}")
        traceback.print_exc()
        return None

    if query_embedding is None:
         print("Failed to retrieve query embedding.")
         return None

    # --- Calculate Similarities ---
    try:
        # Align embeddings and text data based on their common indices before calculating similarity
        common_indices = embeddings_df.index.intersection(text_df.index)
        if len(common_indices) < 2: # Need at least query and one other
             print("Error: Not enough common indices between embeddings and text data after alignment.")
             return None

        aligned_embeddings_values = embeddings_df.loc[common_indices].values
        aligned_text_df = text_df.loc[common_indices] # Use aligned text df for lookups

        # Calculate cosine similarities between the single query embedding and all aligned embeddings
        similarities = cosine_similarity(query_embedding, aligned_embeddings_values)[0]

        # Create a DataFrame to hold similarities, indexed by the common indices
        sim_df = pd.DataFrame({'similarity': similarities}, index=common_indices)
        sim_df = sim_df.sort_values(by='similarity', ascending=False)

        # Exclude the query item itself using its actual index label
        top_similar = sim_df.drop(query_actual_index, errors='ignore').head(top_n)

        if top_similar.empty:
            print("No similar items found (excluding self).")
            return None

        # Retrieve details for the top N similar items
        top_indices = top_similar.index
        top_texts = aligned_text_df.loc[top_indices, text_column].values

        similar_texts_df = pd.DataFrame({
            'index': top_indices, # This will be ID or position depending on common_indices type
            'text': top_texts,
            'similarity': top_similar['similarity'].values
        })

        print(f"\nQuery Text ({query_actual_index}): {query_text[:150]}...")
        print("\nSimilar Texts Found:")
        print(similar_texts_df.to_string()) # Display full dataframe without truncation
        return similar_texts_df

    except Exception as e:
        print(f"Error finding similar texts: {e}")
        traceback.print_exc()
        return None


def analyze_embedding_clusters(
    embeddings_df: pd.DataFrame,
    processed_df: pd.DataFrame, # Can have 'id' or RangeIndex
    tag_columns: list,
    n_clusters: int,
    output_dir: str,
    random_state: int = 42,
    text_column: str = 'cleaned_text_for_lstm',
    n_top_keywords: int = 20,
    perplexity: int = 30
):
    """
    Performs K-Means clustering on embeddings, generates a t-SNE visualization
    colored by cluster, calculates cluster quality metrics, and analyzes the
    content (top tags, top keywords using TF-IDF, word clouds) of each cluster.
    Handles ID or RangeIndex alignment.
    """
    print(f"\n--- Analyzing Embedding Clusters (k={n_clusters}) ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Align Data ---
    print("Aligning embeddings and processed data for clustering...")
    common_indices = embeddings_df.index.intersection(processed_df.index)
    if len(common_indices) == 0:
        print("Error: No common indices found between embeddings and processed data.")
        return None

    embeddings_aligned = embeddings_df.loc[common_indices]
    processed_aligned = processed_df.loc[common_indices].copy()
    print(f"Data aligned. Using {len(common_indices)} common samples.")

    embeddings = embeddings_aligned.values
    n_clusters_adj = min(n_clusters, len(embeddings))
    if n_clusters_adj < 2:
        print(f"Warning: Too few samples ({len(embeddings)}) for K-Means k={n_clusters}. Skipping cluster analysis.")
        return None

    # --- Perform K-Means Clustering ---
    print(f"Running K-Means with k={n_clusters_adj}...")
    kmeans = KMeans(n_clusters=n_clusters_adj, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    processed_aligned['cluster'] = cluster_labels

    # --- Evaluate Cluster Quality (Numerical) ---
    print("\nCalculating Cluster Quality Metrics...")
    try:
        # Silhouette Score requires >= 2 distinct cluster labels
        if len(set(cluster_labels)) >= 2:
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            print(f"  Silhouette Score: {silhouette_avg:.4f} (Higher is better, range -1 to 1)")
        else:
            print("  Silhouette Score: Not applicable (only 1 cluster found or assigned).")

        # Davies-Bouldin Score requires >= 2 distinct cluster labels
        if len(set(cluster_labels)) >= 2:
             db_score = davies_bouldin_score(embeddings, cluster_labels)
             print(f"  Davies-Bouldin Score: {db_score:.4f} (Lower is better, >= 0)")
        else:
             print("  Davies-Bouldin Score: Not applicable (only 1 cluster found or assigned).")

    except Exception as e:
        print(f"  Error calculating cluster metrics: {e}")


    # --- Visualize Clusters (t-SNE) ---
    print("\nGenerating t-SNE plot colored by cluster...")
    tsne_plot_path = os.path.join(output_dir, f"tsne_clusters_k{n_clusters_adj}.png")
    try:
        current_perplexity = min(perplexity, len(embeddings) - 1)
        if current_perplexity <= 0:
             print(f"Warning: Too few samples ({len(embeddings)}) for t-SNE perplexity={perplexity}. Skipping plot.")
        else:
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=current_perplexity, n_jobs=-1, init='pca', learning_rate='auto')
            reduced_embeddings_tsne = tsne.fit_transform(embeddings)

            plot_df = pd.DataFrame({
                'x': reduced_embeddings_tsne[:, 0],
                'y': reduced_embeddings_tsne[:, 1],
                'cluster': cluster_labels
            })

            plt.figure(figsize=(12, 10), dpi=100)
            sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', palette='viridis',
                            alpha=0.6, s=15, legend='full') 
            plt.title(f't-SNE Visualization of Clusters (k={n_clusters_adj})')
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            # Place legend outside plot if too many clusters to avoid overlap
            if n_clusters_adj > 10:
                 plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                 plt.legend(title='Cluster')
                 plt.tight_layout()

            plt.savefig(tsne_plot_path)
            plt.close()
            print(f"  -> t-SNE cluster plot saved to {tsne_plot_path}")

    except Exception as e:
        print(f"  Error generating t-SNE plot: {e}")
        traceback.print_exc()

    # --- Analyze Cluster Content (Qualitative) ---
    print("\n--- Cluster Content Analysis Results ---")
    cluster_summary = {}

    for i in range(n_clusters_adj):
        print(f"\n===== Cluster {i} =====")
        cluster_df = processed_aligned[processed_aligned['cluster'] == i]
        cluster_size = len(cluster_df)
        print(f"Size: {cluster_size} recipes")
        if cluster_size == 0: continue

        cluster_summary[i] = {'size': cluster_size}

        # 1. Top Tags Analysis
        if tag_columns and all(col in cluster_df.columns for col in tag_columns):
            # Sum the binary tag columns within the cluster
            cluster_tag_counts = cluster_df[tag_columns].sum().sort_values(ascending=False)
            top_tags = cluster_tag_counts.head(10)
            print("\nTop 10 Tags (Count | % of Cluster):")
            for tag, count in top_tags.items():
                 # Only display tags actually present in the cluster
                 if count > 0: print(f"  - {tag.replace('tag_', '')}: {count} ({count/cluster_size:.1%})")
            cluster_summary[i]['top_tags'] = top_tags.to_dict()
        else: print("\nTag columns not available or missing in cluster data for analysis.")

        # 2. Top Keywords Analysis (TF-IDF)
        try:
            print(f"\nTop {n_top_keywords} Keywords (TF-IDF):")
            corpus = cluster_df[text_column].astype(str).tolist()
            if not corpus:
                 print("  No text data found in this cluster.")
                 continue

            # TF-IDF Vectorization within the cluster
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.8, min_df=max(2, int(cluster_size*0.01)))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            # Sum TF-IDF scores across all documents in the cluster for each term
            summed_tfidf = tfidf_matrix.sum(axis=0)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.array(summed_tfidf).flatten()
            # Get indices of terms sorted by summed TF-IDF score
            sorted_indices = np.argsort(scores)[::-1]

            top_keywords = {}
            for idx in sorted_indices[:n_top_keywords]:
                 keyword = feature_names[idx]
                 score = scores[idx]
                 print(f"  - {keyword}: {score:.2f}")
                 top_keywords[keyword] = score
            cluster_summary[i]['top_keywords'] = top_keywords

            # 3. Generate Word Cloud
            wordcloud_path = os.path.join(output_dir, f"cluster_{i}_wordcloud.png")
            try:
                 # Use TF-IDF scores as frequencies for the word cloud
                 word_freq = {feature_names[idx]: scores[idx] for idx in sorted_indices[:100] if scores[idx] > 0.1} # Use top 100 words with some min score
                 if word_freq:
                      wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
                      plt.figure(figsize=(15, 7))
                      plt.imshow(wc, interpolation='bilinear')
                      plt.axis('off')
                      plt.title(f"Cluster {i} Top Keywords")
                      plt.savefig(wordcloud_path)
                      plt.close()
                      print(f"  -> Word cloud saved to {wordcloud_path}")
                 else: print("  -> No significant words found for word cloud.")
            except Exception as wc_e: print(f"  -> Error generating word cloud: {wc_e}")

        except Exception as tfidf_e:
            print(f"  Error during TF-IDF analysis: {tfidf_e}")
            traceback.print_exc()

    print("\n--- Cluster Analysis Complete ---")
    return cluster_summary

# --- Utility Functions ---

def visualize_embeddings(embeddings_df, output_dir="results/visualizations", n_clusters=5, random_state=42):
    """Visualize general embedding structure using t-SNE and K-means clustering."""
    print("\nVisualizing general embedding structure (t-SNE + K-Means)...")
    os.makedirs(output_dir, exist_ok=True)
    filepath=os.path.join(output_dir, "embeddings_tsne_kmeans.png")

    embeddings = embeddings_df.values
    # Adjust perplexity based on dataset size, minimum value of 5
    perplexity = min(30, max(5, len(embeddings) - 1))
    if len(embeddings) <= perplexity or len(embeddings) < 2:
        print(f"Warning: Cannot run t-SNE, dataset too small ({len(embeddings)} samples).")
        return None

    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_jobs=-1, init='pca', learning_rate='auto')
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Adjust k for K-Means based on dataset size
    n_clusters_adj = min(n_clusters, len(embeddings))
    if n_clusters_adj < 2:
        print(f"Warning: Skipping K-Means coloring, too few samples ({len(embeddings)}) for k={n_clusters}.")
        clusters = np.zeros(len(embeddings), dtype=int)
    else:
        print(f"Applying K-Means (k={n_clusters_adj})...")
        kmeans = KMeans(n_clusters=n_clusters_adj, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': clusters
    })

    plt.figure(figsize=(10, 8), dpi=100)
    sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster' if n_clusters_adj >= 2 else None,
                    palette='viridis' if n_clusters_adj >= 2 else None,
                    alpha=0.7, s=20, legend='full' if n_clusters_adj >= 2 else None)
    plt.title(f't-SNE visualization of Text Embeddings' + (f' (k={n_clusters_adj} clusters)' if n_clusters_adj >= 2 else ''))
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if n_clusters_adj >= 2:
        plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved general t-SNE/K-Means plot to '{filepath}'")

    return plot_df