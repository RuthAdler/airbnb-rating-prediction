"""
Training script for AirBnB model with GenAI features.

This script:
1. Loads training data (LA + NYC)
2. Preprocesses with original features
3. Adds GenAI features (embeddings + optional LLM)
4. Trains a new model
5. Saves everything needed for inference

Usage:
    # With embeddings only (free, fast)
    python train_genai.py --data-dir data/raw --no-llm
    
    # With embeddings + LLM features
    python train_genai.py --data-dir data/raw --llm-api-key YOUR_KEY
    
    # With custom API endpoint (Nebius)
    python train_genai.py --data-dir data/raw --llm-api-base https://your-endpoint
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import our modules
from src.preprocessing import preprocess_data
from src.genai_features import GenAIFeatureExtractor


def load_training_data(data_dir: str) -> pd.DataFrame:
    """Load and combine LA + NYC training datasets."""
    data_path = Path(data_dir)
    datasets = []
    
    for csv_file in data_path.glob('listings*.csv'):
        if 'TEST' in csv_file.name.upper():
            continue  # Skip test files
            
        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Extract city name
        city = csv_file.stem.replace('listings', '').replace('_', ' ').strip()
        df['city'] = city if city else 'Unknown'
        
        datasets.append(df)
        print(f"  Loaded {len(df)} rows")
    
    if not datasets:
        raise FileNotFoundError(f"No training CSV files found in {data_dir}")
    
    combined = pd.concat(datasets, ignore_index=True)
    print(f"\nTotal training data: {len(combined)} rows")
    
    return combined


def train_with_genai(
    data_dir: str = "data/raw",
    output_dir: str = "models",
    use_llm: bool = True,
    llm_api_key: str = None,
    llm_api_base: str = None,
    embedding_components: int = 15
):
    """
    Train model with GenAI features.
    
    Args:
        data_dir: Directory with training CSV files
        output_dir: Directory to save models
        use_llm: Whether to use LLM features
        llm_api_key: API key for LLM
        llm_api_base: API base URL (for Nebius)
        embedding_components: Number of PCA components for embeddings
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== STEP 1: Load data =====
    print("\n" + "="*50)
    print("STEP 1: Loading training data")
    print("="*50)
    
    df = load_training_data(data_dir)
    df_original = df.copy()  # Keep for GenAI extraction
    
    # ===== STEP 2: Original preprocessing =====
    print("\n" + "="*50)
    print("STEP 2: Original preprocessing")
    print("="*50)
    
    X_train, X_test, y_train, y_test = preprocess_data(df, save_dir=f"{output_dir}/processed")
    
    print(f"Training set: {len(X_train)} rows, {len(X_train.columns)} features")
    print(f"Test set: {len(X_test)} rows")
    
    # ===== STEP 3: GenAI features =====
    print("\n" + "="*50)
    print("STEP 3: Extracting GenAI features")
    print("="*50)
    
    # We need the original text columns for GenAI
    # Get indices that survived preprocessing
    train_indices = X_train.index
    test_indices = X_test.index
    
    df_train_original = df_original.loc[df_original.index.isin(train_indices)].reset_index(drop=True)
    df_test_original = df_original.loc[df_original.index.isin(test_indices)].reset_index(drop=True)
    
    # Initialize GenAI extractor
    genai_extractor = GenAIFeatureExtractor(
        use_embeddings=True,
        use_llm=use_llm,
        embedding_components=embedding_components,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        cache_dir=f"{output_dir}/cache"
    )
    
    # Fit on training data and transform
    print("\nProcessing training set...")
    X_train_genai = genai_extractor.fit_transform(df_train_original)
    
    print("\nProcessing test set...")
    X_test_genai = genai_extractor.transform(df_test_original)
    
    # Save GenAI extractor
    genai_extractor.save(f"{output_dir}/genai_extractor.pkl")
    
    # ===== STEP 4: Combine features =====
    print("\n" + "="*50)
    print("STEP 4: Combining features")
    print("="*50)
    
    # Reset indices for concat
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train_genai = X_train_genai.reset_index(drop=True)
    X_test_genai = X_test_genai.reset_index(drop=True)
    
    X_train_combined = pd.concat([X_train, X_train_genai], axis=1)
    X_test_combined = pd.concat([X_test, X_test_genai], axis=1)
    
    print(f"Combined training features: {len(X_train_combined.columns)}")
    print(f"Feature columns: {list(X_train_combined.columns)}")
    
    # Save feature columns for inference
    feature_columns = list(X_train_combined.columns)
    joblib.dump(feature_columns, f"{output_dir}/feature_columns_genai.pkl")
    
    # ===== STEP 5: Scale features =====
    print("\n" + "="*50)
    print("STEP 5: Scaling features")
    print("="*50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
    joblib.dump(scaler, f"{output_dir}/scaler_genai.pkl")
    print("Scaler saved")
    
    # ===== STEP 6: Train model =====
    print("\n" + "="*50)
    print("STEP 6: Training model")
    print("="*50)
    
    # Try different models
    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Ridge': Ridge(alpha=1.0)
    }
    
    best_model = None
    best_rmse = float('inf')
    best_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='neg_root_mean_squared_error'
        )
        cv_rmse = -cv_scores.mean()
        
        print(f"  CV RMSE: {cv_rmse:.4f} (+/- {cv_scores.std():.4f})")
        
        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (CV RMSE: {best_rmse:.4f})")
    
    # Train final model on all training data
    best_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    # ===== STEP 7: Save model =====
    print("\n" + "="*50)
    print("STEP 7: Saving model")
    print("="*50)
    
    model_path = f"{output_dir}/model_genai.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save summary
    summary = {
        'model_type': best_name,
        'cv_rmse': best_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_features': len(feature_columns),
        'n_original_features': len(X_train.columns),
        'n_genai_features': len(X_train_genai.columns),
        'use_llm': use_llm,
        'embedding_components': embedding_components,
    }
    joblib.dump(summary, f"{output_dir}/training_summary_genai.pkl")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"\nSaved files:")
    print(f"  - {output_dir}/model_genai.pkl")
    print(f"  - {output_dir}/scaler_genai.pkl")
    print(f"  - {output_dir}/genai_extractor.pkl")
    print(f"  - {output_dir}/feature_columns_genai.pkl")
    
    return best_model, scaler, genai_extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AirBnB model with GenAI features")
    
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Directory with training CSV files")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM features (embeddings only)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="API key for LLM (or set OPENAI_API_KEY env var)")
    parser.add_argument("--llm-api-base", type=str, default=None,
                        help="API base URL for LLM (for Nebius/custom endpoints)")
    parser.add_argument("--embedding-components", type=int, default=15,
                        help="Number of PCA components for embeddings")
    
    args = parser.parse_args()
    
    train_with_genai(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
        llm_api_key=args.llm_api_key,
        llm_api_base=args.llm_api_base,
        embedding_components=args.embedding_components
    )
