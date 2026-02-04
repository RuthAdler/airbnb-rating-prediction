"""
Experiment runner with Weights & Biases tracking.
Usage:
    python run_experiment.py --team_member "Ruth" --model random_forest --scaler standard
"""

import argparse
import warnings
import pandas as pd
import numpy as np
import wandb

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loading import load_all_listings
from src.preprocessing import preprocess_data
from src.feature_sets import apply_feature_set

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not installed. Install with: pip install xgboost")


def get_model(config):
    """Return a model instance based on config."""
    model_name = config.model

    if model_name == "dummy":
        return DummyRegressor(strategy="mean")

    elif model_name == "linear_regression":
        return LinearRegression()

    elif model_name == "ridge":
        return Ridge(alpha=config.alpha, random_state=config.random_state)

    elif model_name == "lasso":
        return Lasso(alpha=config.alpha, random_state=config.random_state)

    elif model_name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=config.max_depth,
            random_state=config.random_state
        )

    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1
        )

    elif model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost not installed. Use: pip install xgboost")
        return XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_scaler(scaler_name):
    """Return a scaler instance based on name."""
    if scaler_name == "none":
        return None
    elif scaler_name == "standard":
        return StandardScaler()
    elif scaler_name == "robust":
        return RobustScaler()
    elif scaler_name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")


def evaluate(y_true, y_pred):
    """Calculate evaluation metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred)
    }


def create_run_name(args):
    """Create a descriptive run name based on arguments."""
    run_name = f"{args.team_member}_{args.model}"
    if args.model in ["ridge", "lasso"]:
        run_name += f"_alpha{args.alpha}"
    if args.model in ["decision_tree", "random_forest", "xgboost"] and args.max_depth:
        run_name += f"_depth{args.max_depth}"
    if args.model in ["random_forest", "xgboost"]:
        run_name += f"_n{args.n_estimators}"
    if args.model == "xgboost":
        run_name += f"_lr{args.learning_rate}"
    run_name += f"_{args.scaler}"
    run_name += f"_{args.dataset_version}"
    return run_name


def initialize_wandb(args, run_name):
    """Initialize Weights & Biases run."""
    run = wandb.init(
        project="airbnb-rating-prediction",
        name=run_name,
        config={
            "team_member": args.team_member,
            "dataset_version": args.dataset_version,
            "features": args.dataset_version,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "model": args.model,
            "scaler": args.scaler,
            "alpha": args.alpha,
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
        },
        tags=[args.team_member, args.model, args.scaler, args.dataset_version],
        notes=args.notes
    )
    return run


def print_experiment_header(run):
    """Print experiment information header."""
    config = wandb.config
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {run.name}")
    print(f"Team member: {config.team_member}")
    print(f"Model: {config.model}")
    print(f"Scaler: {config.scaler}")
    print(f"{'=' * 60}\n")


def load_and_preprocess(args):
    """Load and preprocess data."""
    print("Loading data...")
    datasets = load_all_listings("data")
    df = pd.concat(datasets.values(), ignore_index=True)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    return X_train, X_test, y_train, y_test


def clean_data(X_train, X_test):
    """Clean data by handling NaN/inf and clipping extreme values."""
    for col in X_train.columns:
        p99_train = X_train[col].quantile(0.99)
        p1_train = X_train[col].quantile(0.01)
        X_train[col] = X_train[col].clip(lower=p1_train, upper=p99_train)
        X_test[col] = X_test[col].clip(lower=p1_train, upper=p99_train)

    return X_train, X_test


def scale_features(X_train, X_test, scaler_name):
    """Apply scaling to features."""
    scaler = get_scaler(scaler_name)
    if scaler is not None:
        print(f"Applying {scaler_name} scaler...")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    return X_train_scaled, X_test_scaled


def train_and_predict(model, X_train_scaled, X_test_scaled, y_train):
    """Train model and generate predictions."""
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    return y_train_pred, y_test_pred


def print_results(train_metrics, test_metrics, run):
    """Print evaluation results."""
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Train RMSE: {train_metrics['RMSE']:.4f}")
    print(f"Test RMSE:  {test_metrics['RMSE']:.4f}")
    print(f"Train MAE:  {train_metrics['MAE']:.4f}")
    print(f"Test MAE:   {test_metrics['MAE']:.4f}")
    print(f"{'=' * 60}")
    print(f"\nView run at: {run.url}")


def log_feature_importances(model, X_train):
    """Log feature importances for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns.tolist()
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        wandb.log({
            "feature_importances": wandb.Table(dataframe=importance_df.head(20))
        })

        print("\nTop 10 Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


def log_linear_coefs_to_wandb(model, feature_names, top_k=30):
    """Log linear model coefficients to W&B."""
    if not hasattr(model, "coef_"):
        return

    coefs = np.asarray(model.coef_).ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    wandb.log({
        "top_feature": df.iloc[0]["feature"],
        "top_abs_coef": float(df.iloc[0]["abs_coef"]),
        "coefs_table": wandb.Table(dataframe=df.head(top_k)),
    })


def run_experiment(args):
    """Run a single experiment with W&B tracking."""
    run_name = create_run_name(args)
    run = initialize_wandb(args, run_name)
    print_experiment_header(run)

    X_train, X_test, y_train, y_test = load_and_preprocess(args)

    # Apply dataset version feature set
    X_train = apply_feature_set(X_train, args.dataset_version)
    X_test = apply_feature_set(X_test, args.dataset_version)

    # Log dataset info + features (before cleaning/scaling)
    feature_names = X_train.columns.tolist()

    wandb.log({
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_names),
        "features_preview": ", ".join(feature_names),
    })

    with open("features_used.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    wandb.save("features_used.txt")

    # Clean (NaN/inf/clip) and continue
    X_train, X_test = clean_data(X_train, X_test)

    config = wandb.config
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, config.scaler)

    print(f"Training {config.model} model...")
    model = get_model(config)
    y_train_pred, y_test_pred = train_and_predict(model, X_train_scaled, X_test_scaled, y_train)

    if wandb.config.model in {"linear_regression", "ridge", "lasso"}:
        log_linear_coefs_to_wandb(model, X_train.columns.tolist())

    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)

    wandb.log({
        "train_MAE": train_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "train_MSE": train_metrics["MSE"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MSE": test_metrics["MSE"],
    })

    print_results(train_metrics, test_metrics, run)
    log_feature_importances(model, X_train)

    wandb.finish()
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Run ML experiment with W&B tracking")

    # Required
    parser.add_argument("--team_member", type=str, required=True,
                        help="Your name (e.g., 'Ruth')")

    # Model selection
    parser.add_argument("--model", type=str, default="dummy",
                        choices=["dummy", "linear_regression", "ridge", "lasso",
                                 "decision_tree", "random_forest", "xgboost"],
                        help="Model type to use")

    # Preprocessing
    parser.add_argument("--scaler", type=str, default="standard",
                        choices=["none", "standard", "robust", "minmax"],
                        help="Scaler to use")

    # Data config
    parser.add_argument("--dataset_version", type=str, default="v1",
                        help="Dataset version identifier")
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Test set size (0.0-1.0)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")

    # Model hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Regularization strength for Ridge/Lasso")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Max depth for tree models")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees for ensemble models")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate for XGBoost")

    # Optional
    parser.add_argument("--notes", type=str, default="",
                        help="Notes about this experiment")

    args = parser.parse_args()

    # Run the experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
