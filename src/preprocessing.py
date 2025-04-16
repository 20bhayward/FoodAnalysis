# src/preprocessing.py

import pandas as pd
import numpy as np
import re
import ast  # Use ast.literal_eval for safer parsing than eval
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning) # Suppress specific pandas warning

# --- Configuration ---
N_TOP_TAGS = 5 # Default number of most frequent tags to one-hot encode (can be overridden)

# Define the cleaning components globally for use in the function
MEASURES = ['teaspoon', 't','tsp','tablespoon','T','tbsp','cup','c','pint','p','quart','q','gallon','g','oz','ounce','fl','fluid','lb','pound','mg','milligram','gram','kg','kilogram']
WORDS_TO_REMOVE = ['fresh', 'oil', 'a','red','bunch','green','white','black','yellow','large','small','medium','diced',
                   'chopped','sliced','minced','crushed','grated','peeled','seeded','cooked','uncooked','whole','halved',
                   'quartered','cubed','shredded','drained','rinsed','trimmed','divided','beaten','softened','melted','packed',
                   'dried','to','taste','for','serving','optional','as','needed','and','more','or','less','cut','into','strips',''
                   'lengthwise','crosswise','thinly','thickly','sliced','canned','frozen','thawed','at','room','temperature','temp']
# Combine predefined lists and standard English stop words
REMOVE_SET = set(MEASURES + WORDS_TO_REMOVE).union(ENGLISH_STOP_WORDS)

# Expected nutrition column order (must match the order in the raw data string)
NUTRITION_COLS = ['calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv']
NUMERIC_FEATURES = ['minutes', 'n_steps', 'n_ingredients']

def clean_text_for_lstm(text):
    """Cleans text specifically for LSTM input (lowercase, remove non-alpha, stopwords, measures, common words)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    # Remove stop words, measures, common cooking words, and single letters
    cleaned_tokens = [t for t in tokens if t not in REMOVE_SET and len(t) > 1]
    # Join back into a single string
    return " ".join(cleaned_tokens)

def parse_nutrition(nutrition_str):
    """Parses the nutrition string '[val1, val2, ...]' into a pandas Series using expected order."""
    try:
        # Use ast.literal_eval for safety instead of eval
        nutr_list = ast.literal_eval(nutrition_str)
        if isinstance(nutr_list, list) and len(nutr_list) == len(NUTRITION_COLS):
            # Return Series with predefined column names
            return pd.Series(nutr_list, index=NUTRITION_COLS)
        else:
            # Return NaNs if format is unexpected (wrong length or type)
            return pd.Series([np.nan] * len(NUTRITION_COLS), index=NUTRITION_COLS)
    except (ValueError, SyntaxError, TypeError):
        # Handle cases where parsing fails or input is not string/list
         return pd.Series([np.nan] * len(NUTRITION_COLS), index=NUTRITION_COLS)

def parse_tags(tags_str):
    """Parses the tags string "['tag1', 'tag2', ...]" into a list."""
    try:
        tags_list = ast.literal_eval(tags_str)
        if isinstance(tags_list, list):
            return tags_list
        else:
            # Return empty list if format is unexpected
            return []
    except (ValueError, SyntaxError, TypeError):
        # Handle cases where parsing fails or input is not string/list
        return []

def preprocess_data(raw_recipes_path, raw_interactions_path, output_path=None):
    """
    Loads raw recipe and interaction data, performs cleaning, feature engineering,
    selects relevant columns, and returns the processed DataFrame.
    Optionally saves the processed data to CSV.

    Args:
        raw_recipes_path (str): Path to the RAW_recipes.csv file.
        raw_interactions_path (str): Path to the RAW_interactions.csv file.
        output_path (str, optional): Path to save the processed DataFrame as a CSV.
                                     If None, the DataFrame is not saved. Defaults to None.

    Returns:
        pandas.DataFrame: The processed DataFrame with selected features, or None if loading fails.
    """
    # --- 1. Load Data ---
    print("Loading raw data...")
    try:
        data_recipes = pd.read_csv(raw_recipes_path)
        data_interactions = pd.read_csv(raw_interactions_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure raw data files are in the specified paths.")
        return None

    print(f"Loaded {len(data_recipes)} recipes and {len(data_interactions)} interactions.")

    # --- 2. Calculate Average Rating ---
    print("Calculating average ratings...")
    # Ensure rating is numeric, drop invalid entries
    data_interactions['rating'] = pd.to_numeric(data_interactions['rating'], errors='coerce')
    data_interactions.dropna(subset=['rating'], inplace=True)

    # Calculate mean rating per recipe
    avg_rating = data_interactions.groupby('recipe_id')['rating'].mean().reset_index()
    avg_rating.columns = ['id', 'avg_rating'] # Rename columns for merging

    # Merge average rating with recipes data
    recipes_df = data_recipes.merge(avg_rating, on='id', how='left')
    # Keep only recipes that have at least one rating
    recipes_df.dropna(subset=['avg_rating'], inplace=True)
    print(f"{len(recipes_df)} recipes remaining after merging ratings and dropping recipes with no ratings.")

    # --- 3. Combine Text Fields ---
    print("Combining name and description fields...")
    recipes_df['name'] = recipes_df['name'].fillna('')
    recipes_df['description'] = recipes_df['description'].fillna('')
    recipes_df['combined_text'] = recipes_df['name'] + ' ' + recipes_df['description']

    # --- 4. Clean Text for LSTM/Embeddings ---
    print("Cleaning combined text for embedding models...")
    tqdm.pandas(desc="Cleaning Text")
    # Apply the specific cleaning function
    recipes_df['cleaned_text_for_lstm'] = recipes_df['combined_text'].progress_apply(clean_text_for_lstm)

    # --- 5. Extract and Impute Numeric Features ---
    print("Extracting and imputing numeric features (minutes, steps, ingredients)...")
    # Basic NaN handling: fill with median of the column
    for col in NUMERIC_FEATURES:
        median_val = recipes_df[col].median()
        recipes_df[col].fillna(median_val, inplace=True)
        # Ensure type is appropriate (e.g., int or float)
        recipes_df[col] = pd.to_numeric(recipes_df[col], errors='coerce').fillna(median_val)


    # --- 6. Extract and Impute Nutrition Features ---
    print("Extracting and imputing nutrition features...")
    # Apply parsing function to the 'nutrition' column
    nutrition_df = recipes_df['nutrition'].apply(parse_nutrition)

    # Join parsed nutrition columns back with the main dataframe
    recipes_df = pd.concat([recipes_df, nutrition_df], axis=1)

    # Handle potential NaNs introduced during parsing (fill with median)
    for col in NUTRITION_COLS:
        median_val = recipes_df[col].median()
        recipes_df[col].fillna(median_val, inplace=True)
        recipes_df[col] = pd.to_numeric(recipes_df[col], errors='coerce').fillna(median_val)

    # --- 7. Extract Date Features ---
    print("Extracting date features (year, month)...")
    recipes_df['submitted'] = pd.to_datetime(recipes_df['submitted'], errors='coerce')
    recipes_df['submission_year'] = recipes_df['submitted'].dt.year
    recipes_df['submission_month'] = recipes_df['submitted'].dt.month
    # Fill NaNs for year/month (e.g., with median year/month if any dates failed conversion)
    recipes_df['submission_year'].fillna(recipes_df['submission_year'].median(), inplace=True)
    recipes_df['submission_month'].fillna(recipes_df['submission_month'].median(), inplace=True)
    # Convert to integer types after filling NaNs
    recipes_df['submission_year'] = recipes_df['submission_year'].astype(int)
    recipes_df['submission_month'] = recipes_df['submission_month'].astype(int)


    # --- 8. Process Tags (Identify Top N and One-Hot Encode) ---
    # Note: N_TOP_TAGS can be overridden by the pipeline running this function
    print(f"Processing tags (identifying and one-hot encoding top {N_TOP_TAGS})...")
    recipes_df['tags_list'] = recipes_df['tags'].apply(parse_tags)

    # Count all tags across all recipes
    all_tags = Counter(tag for tags in recipes_df['tags_list'] for tag in tags)
    print(f"Found {len(all_tags)} unique tags.")

    # Get the most common tags based on the current N_TOP_TAGS setting
    top_tags = [tag for tag, count in all_tags.most_common(N_TOP_TAGS)]
    print(f"Selected Top {N_TOP_TAGS} tags: {top_tags[:10]}..." if N_TOP_TAGS > 10 else f"Selected Top {N_TOP_TAGS} tags: {top_tags}")

    # Create binary (one-hot) columns for the selected top tags
    tag_categorical_cols = []
    for tag in tqdm(top_tags, desc="Creating Tag Columns"):
        # Sanitize tag name for use as a column name (replace non-alphanumeric with _)
        col_name = f'tag_{re.sub(r"[^a-z0-9_]+", "_", tag.lower())}'
        # Assign 1 if tag is in the recipe's tag list, 0 otherwise
        recipes_df[col_name] = recipes_df['tags_list'].apply(lambda x: 1 if isinstance(x, list) and tag in x else 0)
        tag_categorical_cols.append(col_name)

    # --- 9. Select Final Columns for Processed DataFrame ---
    print("Selecting final columns for the processed dataset...")
    # Define column groups
    id_col = ['id'] # Keep original ID
    target_col = ['avg_rating'] # Include target variable if needed later
    lstm_text_col = ['cleaned_text_for_lstm'] # Main text input for embeddings
    base_numeric_cols = NUMERIC_FEATURES # ['minutes', 'n_steps', 'n_ingredients']
    nutrition_numeric_cols = NUTRITION_COLS # Parsed nutrition values
    date_numeric_cols = ['submission_year', 'submission_month']

    # Combine all desired column names into a single list
    final_columns = (id_col + target_col + lstm_text_col +
                     base_numeric_cols + nutrition_numeric_cols +
                     date_numeric_cols + tag_categorical_cols)

    # Ensure all selected columns actually exist in the DataFrame before selection
    final_columns = [col for col in final_columns if col in recipes_df.columns]
    processed_df = recipes_df[final_columns].copy()

    # --- 10. Final Cleanup ---
    # Drop rows where text cleaning might have resulted in empty strings (important for embeddings)
    initial_len = len(processed_df)
    processed_df = processed_df[processed_df['cleaned_text_for_lstm'].str.strip() != ""]
    if len(processed_df) < initial_len:
        print(f"Dropped {initial_len - len(processed_df)} rows with empty cleaned text.")

    # Reset index after any row drops to ensure it's contiguous
    processed_df.reset_index(drop=True, inplace=True)

    print("\nPreprocessing Complete. Final DataFrame Info:")
    processed_df.info()
    print("\nFinal DataFrame Head:")
    print(processed_df.head())
    print(f"\nShape of final DataFrame: {processed_df.shape}")

    # --- 11. Save the processed DataFrame ---
    if output_path:
        print(f"\nSaving processed data to {output_path}...")
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_df.to_csv(output_path, index=False)
            print(f"Successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

    return processed_df

# Allow running preprocessing directly
if __name__ == "__main__":
    # Define default paths relative to where the script might be run from (e.g., project root)
    DEFAULT_RAW_RECIPES_PATH = os.path.join("data", "raw", "RAW_recipes.csv")
    DEFAULT_RAW_INTERACTIONS_PATH = os.path.join("data", "raw", "RAW_interactions.csv")
    DEFAULT_OUTPUT_PATH = os.path.join("data", "processed", "processed_recipes.csv")

    print("Running preprocessing script directly...")
    # Use the default N_TOP_TAGS value defined at the top of the file
    print(f"(Using default N_TOP_TAGS = {N_TOP_TAGS})")
    preprocess_data(
        raw_recipes_path=DEFAULT_RAW_RECIPES_PATH,
        raw_interactions_path=DEFAULT_RAW_INTERACTIONS_PATH,
        output_path=DEFAULT_OUTPUT_PATH
    )
    print("Preprocessing script finished.")