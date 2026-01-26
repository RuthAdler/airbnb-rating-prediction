# Experiment Guide

W&B Project: https://wandb.ai/ruti-adr-nebius/airbnb-rating-prediction

## Setup

1. Pull latest code: `git pull origin main`
2. Install: `pip install -r requirements.txt`
3. Login to W&B: `wandb login`
4. Test: `python run_experiment.py --team_member "YOUR_NAME" --model dummy`

## How to Run

```
python run_experiment.py --team_member "YOUR_NAME" --model MODEL_NAME
```

Parameters:
- `--team_member` - Your name (required)
- `--model` - dummy, linear_regression, ridge, lasso, decision_tree, random_forest, xgboost
- `--scaler` - none, standard, robust, minmax (default: standard)
- `--max_depth` - Tree depth limit
- `--n_estimators` - Number of trees (default: 100)
- `--alpha` - Regularization for ridge/lasso (default: 1.0)
- `--learning_rate` - For xgboost (default: 0.1)
- `--test_size` - Test split ratio (default: 0.25)
- `--random_state` - Random seed (default: 42)
- `--dataset_version` - Version tag for tracking

## Task Division

Each person tests ONE thing while keeping everything else constant.

### Ruth - Models (DONE)

Tested different algorithms. Best result: XGBoost with RMSE 0.4062

Results:
- XGBoost (lr=0.05): 0.4062
- XGBoost (lr=0.1): 0.4064
- Random Forest (depth=5): 0.4134
- Linear Regression: 0.4142
- Ridge: 0.4142
- Decision Tree (depth=5): 0.4171
- Dummy baseline: 0.4352

Key findings:
- XGBoost works best
- Limit tree depth to avoid overfitting
- Random Forest without depth limit overfits badly

### Person 2 - Scalers

Test if scaling matters.

```
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --scaler none
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --scaler standard
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --scaler robust
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --scaler minmax

python run_experiment.py --team_member "YOUR_NAME" --model linear_regression --scaler none
python run_experiment.py --team_member "YOUR_NAME" --model linear_regression --scaler standard
python run_experiment.py --team_member "YOUR_NAME" --model linear_regression --scaler robust
python run_experiment.py --team_member "YOUR_NAME" --model linear_regression --scaler minmax
```

### Person 3 - Features

Test which features matter. Modify preprocessing.py to create different feature sets.

```
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --dataset_version "v1_all"
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --dataset_version "v2_no_text"
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --dataset_version "v3_host_only"
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --dataset_version "v4_property_only"
```

### Person 4 - Validation

Test if results are stable.

```
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --random_state 42
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --random_state 123
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --random_state 456
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --random_state 789

python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --test_size 0.15
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --test_size 0.20
python run_experiment.py --team_member "YOUR_NAME" --model random_forest --max_depth 5 --test_size 0.30
```

