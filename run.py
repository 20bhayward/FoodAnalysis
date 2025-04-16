# NewDesign/run.py

import argparse
import os
import sys
import time
import pandas as pd
import pickle
import traceback # Keep for printing tracebacks on errors

# --- Add src directory to Python path ---
# This allows importing modules from the 'src' directory directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from pipelines import (
        run_classification_pipeline,
        run_autoencoder_pipeline,
        generate_ae_embeddings,
        generate_sbert_embeddings # SBERT embedding generation pipeline
    )
    import preprocessing # Need access to default N_TOP_TAGS
    import visualization # Need access to visualization functions for analysis
    DEFAULT_N_TAGS = preprocessing.N_TOP_TAGS
except ImportError as e:
    print(f"Error importing pipeline functions/modules from src: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run ML Final Project Pipelines & Analysis")

    # --- Core Pipeline Selection ---
    parser.add_argument(
        "--pipeline", type=str, required=False,
        choices=["classification", "preprocess_only", "autoencoder", "generate_ae_embeddings", "generate_sbert_embeddings"],
        help="Name of the main pipeline to run. Choose one."
    )

    # --- Analysis Triggers ---
    parser.add_argument(
        "--analyze", type=str, nargs='+', choices=['similarity', 'clusters'],
        help="Run specific analysis tasks on existing embeddings. Can choose multiple."
    )
    parser.add_argument(
        "--analysis_embedding_path", type=str, default="results/full_text_embeddings.parquet",
        help="Path to the .parquet embedding file to use for --analyze tasks."
    )
    parser.add_argument(
        "--analysis_query_indices", type=int, nargs='+', default=[0, 100, 1000],
        help="[Similarity Analysis] List of recipe indices/IDs to find similar texts for."
    )
    parser.add_argument(
        "--analysis_num_clusters", type=int, default=10,
        help="[Cluster Analysis] Number of clusters (k) for K-Means."
    )
    parser.add_argument(
        "--analysis_top_n_similar", type=int, default=5,
        help="[Similarity Analysis] Number of similar items to show."
    )

    # --- Data Path Arguments ---
    parser.add_argument("--raw_recipes", type=str, default="data/raw/RAW_recipes.csv", help="Path to raw recipes CSV.")
    parser.add_argument("--raw_interactions", type=str, default="data/raw/RAW_interactions.csv", help="Path to raw interactions CSV.")
    parser.add_argument("--processed_data", type=str, default="data/processed/processed_recipes.csv", help="Path to load/save processed data CSV.")

    # --- Embedding Generation Arguments (Shared & Specific) ---
    parser.add_argument("--embedding_output_dir", type=str, default="results", help="Dir for default LSTM embedding files & outputs.")
    parser.add_argument("--embedding_checkpoint_dir", type=str, default="checkpoints/lstm_classifier", help="Dir for default LSTM model checkpoints/vocab.")
    parser.add_argument("--embedding_input_path", type=str, default=None, help="[Classification] Path to pre-generated .parquet embeddings file to use instead of generating new ones (LSTM default).")

    # --- Results/Visualization Arguments ---
    parser.add_argument("--viz_output_dir", type=str, default="results/visualizations", help="Base directory for saving all visualizations.")

    # --- Pipeline Control Arguments ---
    parser.add_argument("--run-preprocessing", action="store_true", help="Force run main preprocessing step before the selected pipeline.")
    parser.add_argument("--slow-mode", action="store_true", help="[Classification] Disable speed optimizations (e.g., use full data for trees, more epochs).")
    parser.add_argument(
        "--k-folds", type=int, default=1,
        help="[Classification] Number of folds for K-Fold CV. Default: 1 (single train/test split)."
    )
    parser.add_argument(
        "--ablation-tags", type=int, nargs='+', default=None,
        help="[Classification] Run classification pipeline multiple times for specified list of 'N_TOP_TAGS' values (triggers preprocessing for each)."
    )

    # --- Autoencoder Specific Arguments ---
    parser.add_argument("--ae_checkpoint_dir", type=str, default="checkpoints/lstm_autoencoder", help="[Autoencoder] Directory for AE checkpoints/vocab.")
    parser.add_argument("--ae_vocab_filename", type=str, default="ae_vocab.pkl", help="[Autoencoder] Filename for the AE vocabulary.")
    parser.add_argument("--ae_epochs", type=int, default=10, help="[Autoencoder] Number of training epochs.")
    parser.add_argument("--ae_batch_size", type=int, default=64, help="[Autoencoder] Batch size for training.")
    parser.add_argument("--ae_embed_dim", type=int, default=100, help="[Autoencoder] Embedding dimension.")
    parser.add_argument("--ae_hidden_dim", type=int, default=128, help="[Autoencoder] LSTM hidden dimension.")
    parser.add_argument("--ae_max_len", type=int, default=100, help="[Autoencoder] Max sequence length for AE training.")
    parser.add_argument("--ae_max_vocab", type=int, default=15000, help="[Autoencoder] Max vocabulary size for AE training.")
    parser.add_argument("--ae_subsample_size", type=int, default=50000, help="[Autoencoder] Number of samples to use for AE training (0 or None for full dataset).")
    parser.add_argument("--ae_embedding_output_path", type=str, default="results/ae_full_text_embeddings.parquet", help="[Generate AE Embeddings] Path to save AE-generated embeddings.")

    # --- Sentence-BERT Specific Arguments ---
    parser.add_argument("--sbert_model_name", type=str, default="all-MiniLM-L6-v2", help="[SBERT] Name of the Sentence Transformer model to use.")
    parser.add_argument("--sbert_batch_size", type=int, default=64, help="[SBERT] Batch size for SBERT encoding.")
    parser.add_argument("--sbert_embedding_output_path", type=str, default="results/sbert_embeddings.parquet", help="[Generate SBERT Embeddings] Path to save SBERT-generated embeddings.")


    args = parser.parse_args()

    # --- Validate Arguments ---
    if not args.pipeline and not args.analyze:
        parser.error("No action requested. Please specify a --pipeline or at least one --analyze task.")
    if args.ablation_tags and args.pipeline != "classification":
        parser.error("--ablation-tags can only be used with --pipeline classification.")

    fast_mode = not args.slow_mode
    overall_start_time = time.time()

    # --- Execute Selected Pipeline (if any) ---
    if args.pipeline:
        print(f"\n>>> Running Pipeline: {args.pipeline} <<<")

        if args.pipeline == "classification":
             # --- Classification Pipeline (Single Run or Ablation Study) ---
             if args.ablation_tags:
                # --- Ablation Study ---
                print(f"*** Running Ablation Study for Top Tags: {args.ablation_tags} ***")
                tag_counts_to_run = sorted([int(t) for t in args.ablation_tags if int(t) > 0])
                if not tag_counts_to_run:
                     print("Error: Invalid values provided for --ablation-tags.")
                     sys.exit(1)

                all_ablation_metrics = {}
                original_n_tags_setting = preprocessing.N_TOP_TAGS # Store original default

                for n_tags in tag_counts_to_run:
                    run_start_time = time.time()
                    print(f"\n===== Running Ablation Configuration: Top {n_tags} Tags =====")
                    # Define unique output paths for this ablation run
                    ablation_suffix = f"_top{n_tags}"
                    current_processed_data_path = args.processed_data.replace(".csv", f"{ablation_suffix}.csv")
                    current_classification_viz_dir = os.path.join(args.viz_output_dir, f"classification{ablation_suffix}")
                    # Embeddings don't depend on tags, but tag viz does
                    current_embedding_tags_viz_dir = os.path.join(args.viz_output_dir, f"embedding_tags{ablation_suffix}")
                    print(f"    (Preprocessing output: {current_processed_data_path})")
                    print(f"    (Classification viz output: {current_classification_viz_dir})")

                    if args.k_folds > 1: print(f"   (Using {args.k_folds}-Fold Cross-Validation)")
                    else: print("   (Using single Train/Test Split)")
                    print(" (Forcing preprocessing for tag ablation)")

                    try:
                        # Run the classification pipeline with overridden N_TOP_TAGS
                        # This implicitly runs preprocessing with the correct n_tags
                        metrics_results = run_classification_pipeline(
                            processed_data_path=current_processed_data_path,
                            embedding_output_dir=args.embedding_output_dir,
                            embedding_checkpoint_dir=args.embedding_checkpoint_dir,
                            embedding_input_path=args.embedding_input_path, # Use same embeddings if provided
                            viz_output_dir=args.viz_output_dir,
                            classification_viz_dir=current_classification_viz_dir,
                            embedding_tags_viz_dir=current_embedding_tags_viz_dir,
                            fast_mode=fast_mode,
                            run_preprocessing=True, # Force preprocessing with override
                            raw_recipes_path=args.raw_recipes,
                            raw_interactions_path=args.raw_interactions,
                            k_folds=args.k_folds,
                            n_top_tags_override=n_tags # Pass the override
                        )
                        all_ablation_metrics[n_tags] = metrics_results # Store results keyed by n_tags
                    except Exception as e:
                        print(f"!!!!! ERROR during classification run (N_TAGS={n_tags}) !!!!!\n{e}")
                        traceback.print_exc() # Print full traceback for debugging

                    run_end_time = time.time()
                    print(f"===== Ablation Configuration Top {n_tags} Finished in {run_end_time - run_start_time:.2f} seconds =====")

                # Restore original N_TOP_TAGS setting in preprocessing module (good practice)
                preprocessing.N_TOP_TAGS = original_n_tags_setting
                print(f"\nRestored preprocessing.N_TOP_TAGS to {original_n_tags_setting}")

                print("\n*** Ablation Study Complete ***")
                # Plot summary results if any metrics were collected
                if all_ablation_metrics:
                     try:
                          print("\n--- Plotting Ablation Summaries ---")
                          # Plot summaries for potentially informative metrics/models
                          visualization.plot_ablation_summary(all_ablation_metrics, args.viz_output_dir, metric_key='f1_macro', model_name='Random Forest')
                          visualization.plot_ablation_summary(all_ablation_metrics, args.viz_output_dir, metric_key='accuracy', model_name='Neural Network')
                          visualization.plot_ablation_summary(all_ablation_metrics, args.viz_output_dir, metric_key='hamming_loss', model_name='Random Forest')
                     except Exception as e: print(f"Error plotting ablation summary: {e}")

             else:
                # --- Single Classification Run ---
                print(f"*** Running Single Classification Pipeline ***")
                if args.embedding_input_path:
                    print(f"--- Using pre-generated embeddings from: {args.embedding_input_path} ---")
                else:
                    print(f"--- Generating/Using default LSTM embeddings ---")
                if args.k_folds > 1: print(f"Using {args.k_folds}-Fold Cross-Validation.")
                else: print("Using single Train/Test Split.")

                # Define output directories for this run
                classification_viz_dir = os.path.join(args.viz_output_dir, "classification")
                # Use embedding type in dir name if specified, otherwise default
                embed_type_suffix = ""
                if args.embedding_input_path:
                    if "sbert" in args.embedding_input_path.lower(): embed_type_suffix = "_sbert"
                    elif "ae_" in args.embedding_input_path.lower(): embed_type_suffix = "_ae"
                embedding_tags_viz_dir = os.path.join(args.viz_output_dir, f"embedding_tags{embed_type_suffix}")

                run_classification_pipeline(
                    processed_data_path=args.processed_data,
                    embedding_output_dir=args.embedding_output_dir,
                    embedding_checkpoint_dir=args.embedding_checkpoint_dir,
                    embedding_input_path=args.embedding_input_path,
                    viz_output_dir=args.viz_output_dir,
                    classification_viz_dir=classification_viz_dir,
                    embedding_tags_viz_dir=embedding_tags_viz_dir,
                    fast_mode=fast_mode,
                    run_preprocessing=args.run_preprocessing,
                    raw_recipes_path=args.raw_recipes,
                    raw_interactions_path=args.raw_interactions,
                    k_folds=args.k_folds,
                    n_top_tags_override=None # No override for single run
                )

        elif args.pipeline == "preprocess_only":
             # --- Preprocessing Only ---
             print(f"*** Running Preprocessing Only ***")
             # Use the first value from ablation tags if provided, else default
             n_tags_to_process = args.ablation_tags[0] if args.ablation_tags else DEFAULT_N_TAGS
             print(f"(Using N_TOP_TAGS = {n_tags_to_process} for this run)")
             original_n_tags_setting = preprocessing.N_TOP_TAGS
             preprocessing.N_TOP_TAGS = n_tags_to_process
             try:
                 preprocessing.preprocess_data(
                      raw_recipes_path=args.raw_recipes,
                      raw_interactions_path=args.raw_interactions,
                      output_path=args.processed_data
                 )
             finally:
                  preprocessing.N_TOP_TAGS = original_n_tags_setting # Restore default

        elif args.pipeline == "autoencoder":
             # --- Autoencoder Training ---
             print(f"*** Running Autoencoder Training Pipeline ***")
             run_autoencoder_pipeline(
                 processed_data_path=args.processed_data,
                 text_column='cleaned_text_for_lstm',
                 ae_checkpoint_dir=args.ae_checkpoint_dir,
                 ae_vocab_filename=args.ae_vocab_filename,
                 ae_checkpoint_filename="lstm_autoencoder_checkpoint.pt", # Default filename
                 max_len=args.ae_max_len,
                 embedding_dim=args.ae_embed_dim,
                 hidden_dim=args.ae_hidden_dim,
                 batch_size=args.ae_batch_size,
                 epochs=args.ae_epochs,
                 learning_rate=1e-3, # Common default LR
                 run_preprocessing=args.run_preprocessing,
                 raw_recipes_path=args.raw_recipes,
                 raw_interactions_path=args.raw_interactions,
                 ae_max_vocab=args.ae_max_vocab,
                 ae_subsample_size=args.ae_subsample_size if args.ae_subsample_size > 0 else None
             )

        elif args.pipeline == "generate_ae_embeddings":
              # --- Generate Embeddings from Trained AE ---
              print(f"*** Running Generate AE Embeddings Pipeline ***")
              generate_ae_embeddings(
                  processed_data_path=args.processed_data,
                  text_column='cleaned_text_for_lstm',
                  ae_checkpoint_dir=args.ae_checkpoint_dir,
                  ae_vocab_filename=args.ae_vocab_filename,
                  ae_checkpoint_filename="lstm_autoencoder_checkpoint.pt", # Default filename
                  output_path=args.ae_embedding_output_path,
                  batch_size=args.ae_batch_size
              )

        elif args.pipeline == "generate_sbert_embeddings":
              # --- Generate Embeddings using SBERT ---
              print(f"*** Running Generate SBERT Embeddings Pipeline ***")
              generate_sbert_embeddings(
                  processed_data_path=args.processed_data,
                  text_column='cleaned_text_for_lstm', # Choose text column appropriate for SBERT
                  model_name=args.sbert_model_name,
                  output_path=args.sbert_embedding_output_path,
                  batch_size=args.sbert_batch_size
              )
        else:
            # This case should not be reached due to argparse choices
            print(f"Error: Unknown pipeline '{args.pipeline}'")
            sys.exit(1)

    # --- Execute Analysis Tasks (if any) ---
    if args.analyze:
        print(f"\n>>> Running Analysis Tasks: {args.analyze} <<<")

        processed_df_analysis = None
        embeddings_df_analysis = None
        tag_columns_analysis = None

        # Load data needed for analysis tasks
        if 'similarity' in args.analyze or 'clusters' in args.analyze:
            print(f"Loading data for analysis...")
            try:
                print(f"  Loading processed data: {args.processed_data}")
                processed_df_analysis = pd.read_csv(args.processed_data)
                # Try setting 'id' as index if it exists and is unique
                if 'id' in processed_df_analysis.columns and processed_df_analysis['id'].is_unique:
                     processed_df_analysis = processed_df_analysis.set_index('id', drop=False)
                     print("  Set unique 'id' column as index for processed_df.")
                else:
                     processed_df_analysis = processed_df_analysis.reset_index(drop=True) # Use default RangeIndex
                     print("  Using default RangeIndex for processed_df (ID missing, not unique, or not set as index).")

                tag_columns_analysis = processed_df_analysis.columns[processed_df_analysis.columns.str.contains('tag_', case=False)].tolist()

            except FileNotFoundError:
                print(f"  ERROR: Processed data file not found at {args.processed_data}. Cannot run analysis.")
                args.analyze = [] # Clear analysis tasks if data is missing
            except Exception as e:
                print(f"  ERROR loading processed data: {e}")
                traceback.print_exc()
                args.analyze = []

            # Load embeddings only if processed data loaded successfully
            if args.analyze:
                try:
                    print(f"  Loading embeddings: {args.analysis_embedding_path}")
                    embeddings_df_analysis = pd.read_parquet(args.analysis_embedding_path)
                    if embeddings_df_analysis.index.name == 'id':
                        print("  Embeddings file has 'id' index.")
                    else:
                        print("  Embeddings file has default RangeIndex.")
                except FileNotFoundError:
                    print(f"  ERROR: Embeddings file not found at {args.analysis_embedding_path}. Cannot run analysis.")
                    args.analyze = []
                except Exception as e:
                    print(f"  ERROR loading embeddings: {e}")
                    traceback.print_exc()
                    args.analyze = []

        # Run selected analyses if data loaded successfully
        if 'similarity' in args.analyze and embeddings_df_analysis is not None and processed_df_analysis is not None:
            print("\n== Running Similarity Analysis ==")
            analysis_output_dir = os.path.join(args.viz_output_dir, "analysis_similarity")
            os.makedirs(analysis_output_dir, exist_ok=True)
            print(f"Output will be printed below (no files saved for similarity).")

            # Determine if query IDs are positional based on embedding index type
            is_positional_query = embeddings_df_analysis.index.name != 'id'
            print(f"Treating query indices as {'positional (0-based)' if is_positional_query else 'recipe IDs based on embedding index'}.")

            for query_val in args.analysis_query_indices:
                visualization.find_similar_texts(
                    query_identifier=query_val,
                    is_positional=is_positional_query,
                    embeddings_df=embeddings_df_analysis,
                    text_df=processed_df_analysis, # Pass df with its own index (ID or RangeIndex)
                    text_column='cleaned_text_for_lstm',
                    top_n=args.analysis_top_n_similar
                )

        if 'clusters' in args.analyze and embeddings_df_analysis is not None and processed_df_analysis is not None:
            print("\n== Running Cluster Analysis ==")
            embed_basename = os.path.splitext(os.path.basename(args.analysis_embedding_path))[0]
            analysis_output_dir = os.path.join(args.viz_output_dir, f"analysis_clusters_{embed_basename}")
            os.makedirs(analysis_output_dir, exist_ok=True)
            print(f"Saving cluster analysis results (plots, word clouds) to: {analysis_output_dir}")

            visualization.analyze_embedding_clusters(
                 embeddings_df=embeddings_df_analysis, # Embeddings (ID or RangeIndex)
                 processed_df=processed_df_analysis, # Processed data (ID or RangeIndex)
                 tag_columns=tag_columns_analysis if tag_columns_analysis else [],
                 n_clusters=args.analysis_num_clusters,
                 output_dir=analysis_output_dir,
                 random_state=42, # Use fixed random state for reproducibility
                 text_column='cleaned_text_for_lstm' # Column for TF-IDF/WordCloud
            )

    overall_end_time = time.time()
    print(f"\nTotal execution finished in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()