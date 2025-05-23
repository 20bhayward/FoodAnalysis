﻿
## Setup Instructions

1.  **Prerequisites:**
    *   Python (>= 3.9 recommended)
    *   `pip` or `conda` for package management

2.  **Clone Repository:**
    ```bash
    git clone [<your-repo-url>](https://github.com/20bhayward/FoodAnalysis.git)
    cd FoodAnalysis
    ```

3.  **Create Virtual Environment (Recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # Linux/macOS
        # venv\Scripts\activate   # Windows
        ```
    *   Using `conda`:
        ```bash
        conda create -n food_ml python=3.10 # Or your preferred version
        conda activate food_ml
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This will install PyTorch, TensorFlow, scikit-learn, pandas, sentence-transformers, etc.)*

5.  **Download Data:**
    *   Download the dataset from [Kaggle: Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).
    *   Place `RAW_recipes.csv` and `RAW_interactions.csv` into the `NewDesign/data/raw/` directory.

## How to Run (`run.py`)

The main entry point for executing the defined pipelines is `run.py`. It uses command-line arguments to control execution.

**Key Pipelines:**

*   `preprocess_only`: Runs only the data preprocessing step defined in `src/preprocessing.py`.
*   `classification`: Runs the full multi-label tag classification pipeline.
*   `autoencoder`: Trains the LSTM Autoencoder model.
*   `generate_ae_embeddings`: Loads a trained AE and generates embeddings for the full dataset.
*   `generate_sbert_embeddings`: Generates Sentence-BERT embeddings for the full dataset.

**Key Arguments:**

*   `--pipeline {classification,preprocess_only,autoencoder,generate_ae_embeddings,generate_sbert_embeddings}`: **(Required)** Specifies which pipeline to run.
*   `--run-preprocessing`: (Flag) Forces the preprocessing step (`src/preprocessing.py`) to run before the main pipeline, overwriting the existing processed data file. Useful when changing preprocessing logic or the number of tags.
*   `--slow-mode`: (Flag, Classification Pipeline) Disables "fast mode" optimizations. This typically means: PCA is disabled, models use potentially slower/more complex settings, and Gradient Boosting is trained. *(Default is Fast Mode: PCA enabled, faster model params, GB skipped)*.
*   `--k-folds N`: (Classification Pipeline) Sets the number of folds for K-Fold Cross-Validation. If `N=1` (default), a single 80/20 train/test split is used.
*   `--ablation-tags N1 N2 ...`: (Classification Pipeline) Runs the classification pipeline multiple times, once for each specified number `N` of top tags (defined in `preprocessing.py`). Automatically forces preprocessing for each run and saves outputs to separate subdirectories (e.g., `results/visualizations/classification_top5/`).
*   `--embedding_input_path PATH`: (Classification Pipeline) Specifies the path to a pre-generated embeddings `.parquet` file (e.g., from AE or SBERT pipelines) to use instead of generating the default LSTM embeddings. The index of this file must align with the processed data's index (usually recipe `id`).
*   `--processed_data PATH`: Specifies the path for the processed CSV file (default: `data/processed/processed_recipes.csv`). Modified automatically during ablation runs.
*   `--embedding_output_dir PATH`: Specifies the base directory for saving embedding `.parquet` files (default: `results`).
*   `--embedding_checkpoint_dir PATH`: Specifies the base directory for saving the classification LSTM embedder model/vocab (default: `checkpoints/lstm_classifier`).
*   `--viz_output_dir PATH`: Base directory for all visualization outputs (default: `results/visualizations`). Specific subdirectories are generated based on the pipeline/ablation run.
*   `--ae_*`: Arguments specific to the `autoencoder` and `generate_ae_embeddings` pipelines (e.g., `--ae_epochs`, `--ae_embed_dim`, `--ae_checkpoint_dir`). Check `run.py --help` for details.
*   `--sbert_*`: Arguments specific to the `generate_sbert_embeddings` pipeline (e.g., `--sbert_model_name`, `--sbert_output_path`). Check `run.py --help` for details.

**Example Commands (Run from project directory):**

1.  **Run Preprocessing Only (using default N\_TOP\_TAGS defined in `preprocessing.py`):**
    ```bash
    python run.py --pipeline preprocess_only
    ```

2.  **Run Classification (Fast Mode, Single Split, using default LSTM embeddings generated on the fly, assuming processed data exists):**
    ```bash
    python run.py --pipeline classification
    ```

3.  **Run Classification (Force Preprocessing, Fast Mode, Single Split):**
    ```bash
    python run.py --pipeline classification --run-preprocessing
    ```

4.  **Run Classification (Slow Mode, 5-Fold CV, using default LSTM embeddings):**
    ```bash
    python run.py --pipeline classification --slow-mode --k-folds 5
    ```

5.  **Run Ablation Study (Top 5 and Top 10 Tags, Fast Mode, Single Split):**
    ```bash
    python run.py --pipeline classification --ablation-tags 5 10
    ```
    *(This will force preprocessing for N=5 and N=10, saving outputs to `_top5` and `_top10` subdirectories respectively)*

6.  **Run Autoencoder Training (Defaults, assuming processed data exists):**
    ```bash
    python run.py --pipeline autoencoder
    ```

7.  **Generate Autoencoder Embeddings (assuming AE model is trained and processed data exists):**
    ```bash
    python run.py --pipeline generate_ae_embeddings --ae_checkpoint_dir checkpoints/lstm_autoencoder --ae_output_path results/ae_embeddings.parquet
    ```

8.  **Generate SBERT Embeddings (using default 'all-MiniLM-L6-v2' model):**
    ```bash
    python run.py --pipeline generate_sbert_embeddings --sbert_output_path results/sbert_embeddings.parquet
    ```

9.  **Run Classification using Pre-generated SBERT Embeddings (Fast Mode, Single Split):**
    ```bash
    python run.py --pipeline classification --embedding_input_path results/sbert_embeddings.parquet
    ```

## Features & Pipelines

*   **Preprocessing (`preprocess_only` pipeline):**
    *   Loads raw recipes and interactions data.
    *   Calculates average ratings and merges datasets.
    *   Combines and cleans text fields specifically for embedding models (`cleaned_text_for_lstm`).
    *   Extracts and imputes numeric features (duration, steps, ingredients, nutrition, date).
    *   Identifies the `N` most frequent tags (configurable via `N_TOP_TAGS` in `preprocessing.py` or `--ablation-tags` in `run.py`).
    *   One-hot encodes these top tags as binary features (`tag_*`).
    *   Saves the final processed DataFrame (including `id` column) to CSV.

*   **Classification (`classification` pipeline):**
    *   Loads processed data.
    *   Generates text embeddings using the LSTM `TextEncoder` by default, or loads pre-generated embeddings from a `.parquet` file using `--embedding_input_path`.
    *   Performs data splitting (Train/Test or K-Fold CV).
    *   Applies `StandardScaler` and optional `PCA` (controlled by `--slow-mode`) to embeddings.
    *   Trains multiple multi-label classification models:
        *   Neural Network (Keras Sequential)
        *   Random Forest (scikit-learn MultiOutputClassifier)
        *   Logistic Regression (scikit-learn MultiOutputClassifier)
        *   Gradient Boosting (scikit-learn MultiOutputClassifier, only if `--slow-mode` is used).
    *   Evaluates models using:
        *   Subset Accuracy (Exact Match Ratio)
        *   Hamming Loss
        *   Macro F1-Score
        *   Per-Tag F1-Score and Accuracy
    *   Generates visualizations:
        *   t-SNE plots of embeddings colored by tags.
        *   Model performance comparison plots.
        *   Per-tag performance heatmaps.
        *   Feature importance plots (for Random Forest).
        *   Learning curves (for Neural Network).
    *   Performs basic error analysis on misclassified examples.
    *   Supports ablation studies on the number of top tags (`--ablation-tags`).

*   **Autoencoder Training (`autoencoder` pipeline):**
    *   Loads processed data.
    *   Uses a custom `RecipeTextDatasetAE` which performs AE-specific text cleaning and builds/uses a separate vocabulary (limited size).
    *   Trains an `LSTMAutoencoder` model (defined in `models.py`).
    *   Saves the trained model state dictionary and the AE vocabulary.

*   **Autoencoder Embedding Generation (`generate_ae_embeddings` pipeline):**
    *   Loads a pre-trained `LSTMAutoencoder` model and its vocabulary.
    *   Loads the processed data.
    *   Uses the *encoder* part of the AE model (`model.encode()`) to generate embeddings for the `cleaned_text_for_lstm` column.
    *   Saves the resulting embeddings (with recipe `id` as index) to a `.parquet` file.

*   **SBERT Embedding Generation (`generate_sbert_embeddings` pipeline):**
    *   Loads processed data.
    *   Uses the `sentence-transformers` library to load a pre-trained SBERT model (e.g., `all-MiniLM-L6-v2`).
    *   Generates embeddings for the `cleaned_text_for_lstm` column.
    *   Saves the resulting embeddings (with recipe `id` as index) to a `.parquet` file.

## Results Artifacts

Running the pipelines (especially `classification`) generates various outputs stored primarily in the `results/` and `checkpoints/` directories:

*   **Processed Data:** `data/processed/processed_recipes.csv` (or similar if ablation is used).
*   **Embeddings:** `.parquet` files in `results/` containing embeddings (e.g., `full_text_embeddings.parquet` for default LSTM, `ae_embeddings.parquet`, `sbert_embeddings.parquet`).
*   **Model Checkpoints:** Saved model states (e.g., `checkpoints/lstm_classifier/text_encoder.pt`, `checkpoints/lstm_autoencoder/lstm_autoencoder_checkpoint.pt`) and vocabularies (`.pkl` files).
*   **Visualizations:** Various plots saved as `.png` files in `results/visualizations/`, organized into subdirectories:
    *   t-SNE embedding plots (colored by tags, clusters).
    *   Model performance comparison bar charts.
    *   Tag performance heatmaps.
    *   Feature importance plots.
    *   Neural Network learning curves.
    *   Cluster analysis plots (t-SNE, word clouds).
    *   Ablation summary plots.
*   **Metrics:** Performance metrics are printed to the console during execution and aggregated in final reports (especially for K-Fold CV).
