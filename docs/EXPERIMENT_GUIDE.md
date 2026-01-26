# Experiment Guide

This document explains how to run experiments for the Airbnb Rating Prediction project using Weights & Biases (W&B).

---

## W&B Project Link

**https://wandb.ai/ruti-adr-nebius/airbnb-rating-prediction**

---

## Setup (Everyone)

### Step 1: Pull latest code
```bash
git checkout main
git pull origin main
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Login to W&B
```bash
wandb login
```
Get your API key from: https://wandb.ai/authorize

### Step 4: Test your setup
```bash
python run_experiment.py --team_member "YOUR_NAME" --model dummy
```
You should see a W&B link in the output. Click it to verify your run appeared.

---

## How to Run Experiments

Basic command:
```bash
python run_experiment.py --team_member "YOUR_NAME" --model MODEL_NAME
```

All available parameters:

| Parameter | Options | Default |
|-----------|---------|---------|
| `--team_member` | Your name | Required |
| `--model` | dummy, linear_regression, ridge, lasso, decision_tree, random_forest, xgboost | dummy |
| `--scaler` | none, standard, robust, minmax | standard |
| `--dataset_version` | any string (for tracking) | v1 |
| `--test_size` | 0.1 to 0.5 | 0.25 |
| `--random_state` | any integer | 42 |
| `--alpha` | any float (for ridge/lasso) | 1.0 |
| `--max_depth` | any integer (for trees) | None |
| `--n_estimators` | any integer (for forests) | 100 |
| `--learning_rate` | any float (for xgboost) | 0.1 |

---

## Task Division

We divided the work so each person tests ONE aspect while keeping everything else constant. This is proper scientific methodology — change one variable at a time.

---

## Ruth — Model Experiments ✅ COMPLETED

**Goal:** Find which algorithm works best for predicting ratings.

**What to test:** Different model types and their hyperparameters.

**What to keep constant:** Scaler (standard), features (all), test_size (0.25), random_state (42).

### Results

| Model | Test RMSE | Train RMSE | Notes |
|-------|-----------|------------|-------|
| **XGBoost (lr=0.05)** | **0.4062** | 0.3808 | 🏆 Best |
| XGBoost (lr=0.1) | 0.4064 | 0.3623 | Very close |
| XGBoost (depth=5, lr=0.1) | 0.4068 | 0.3807 | |
| XGBoost (n=200, lr=0.1) | 0.4094 | 0.3342 | Slight overfit |
| Random Forest (depth=5) | 0.4134 | 0.4113 | Good, no overfit |
| Linear Regression | 0.4142 | 0.4172 | Solid baseline |
| Ridge (alpha=1.0) | 0.4142 | 0.4172 | Same as linear |
| Ridge (alpha=10.0) | 0.4142 | 0.4172 | Same |
| Decision Tree (depth=5) | 0.4171 | 0.4153 | OK |
| Random Forest (n=100) | 0.4188 | 0.1573 | ⚠️ Overfitting |
| Dummy (baseline) | 0.4352 | 0.4393 | Baseline to beat |
| Lasso (alpha=1.0) | 0.4352 | 0.4393 | Too much regularization |
| Decision Tree (depth=10) | 0.4390 | 0.3718 | ⚠️ Overfitting |

### Key Findings
1. **XGBoost performs best** with RMSE 0.4062
2. **Limiting tree depth prevents overfitting** — Random Forest depth=5 works much better than unlimited
3. **Linear models are stable** — no overfitting risk
4. **Lasso with alpha=1.0 is too strong** — kills the model
5. **Default Random Forest overfits badly** — Train 0.16 vs Test 0.42

### Recommendation for Team
Use `random_forest --max_depth 5` as your base model for experiments. It's stable and performs well.

---

## Ido — Scaler Experiments

**Goal:** Find if data scaling matters and which scaler works best.

**What to test:** Different scaling methods (none, standard, robust, minmax).

**What to keep constant:** Features (all), test_size (0.25), random_state (42).

**Use these models:** random_forest (with --max_depth 5) and linear_regression

**Run these experiments:**
```bash
# Random Forest with different scalers
python run_experiment.py --team_member "Ido" --model random_forest --max_depth 5 --scaler none
python run_experiment.py --team_member "Ido" --model random_forest --max_depth 5 --scaler standard
python run_experiment.py --team_member "Ido" --model random_forest --max_depth 5 --scaler robust
python run_experiment.py --team_member "Ido" --model random_forest --max_depth 5 --scaler minmax

# Linear Regression with different scalers
python run_experiment.py --team_member "Ido" --model linear_regression --scaler none
python run_experiment.py --team_member "Ido" --model linear_regression --scaler standard
python run_experiment.py --team_member "Ido" --model linear_regression --scaler robust
python run_experiment.py --team_member "Ido" --model linear_regression --scaler minmax
```

**Question to answer:** Does the scaler choice affect tree models? Does it affect linear models?

---

## Ella — Feature Experiments

**Goal:** Find which features are useful for predicting ratings.

**What to test:** Different feature sets by modifying preprocessing.

**What to keep constant:** Model (random_forest --max_depth 5), scaler (standard), test_size (0.25), random_state (42).

**How to create different feature sets:**
1. Modify `preprocessing.py` to include/exclude certain features
2. Change `--dataset_version` to track which version you used
3. Document what features are in each version

**Run these experiments:**
```bash
# Baseline - all features
python run_experiment.py --team_member "Ella" --model random_forest --max_depth 5 --dataset_version "v1_all"

# After modifying preprocessing.py to remove text features:
python run_experiment.py --team_member "Ella" --model random_forest --max_depth 5 --dataset_version "v2_no_text"

# After modifying to keep only host features:
python run_experiment.py --team_member "Ella" --model random_forest --max_depth 5 --dataset_version "v3_host_only"

# After modifying to keep only property features:
python run_experiment.py --team_member "Ella" --model random_forest --max_depth 5 --dataset_version "v4_property_only"

# After adding geo features (use geo_processing.py):
python run_experiment.py --team_member "Ella" --model random_forest --max_depth 5 --dataset_version "v5_with_geo"
```

**Feature groups to consider:**
- Host features: host_response_rate, host_acceptance_rate, host_is_superhost, etc.
- Property features: bedrooms, bathrooms, beds, accommodates, etc.
- Text features: description_length_words, description_length_chars, etc.
- Geo features: area clusters, distance_to_center (from geo_processing.py)

---

## Rosemary — Validation Experiments

**Goal:** Check if our results are stable and reliable, not just lucky.

**What to test:** Different random seeds and train/test split sizes.

**What to keep constant:** Model (random_forest --max_depth 5), scaler (standard), features (all).

**Run these experiments:**
```bash
# Stability test - same config, different random seeds
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --random_state 42
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --random_state 123
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --random_state 456
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --random_state 789
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --random_state 999

# Test size experiments
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --test_size 0.15
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --test_size 0.20
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --test_size 0.25
python run_experiment.py --team_member "Rosemary" --model random_forest --max_depth 5 --test_size 0.30
```

**Questions to answer:**
- How much does RMSE vary between different random seeds?
- Does test_size affect our results significantly?

**Additional task:** Create a summary dashboard on W&B.

---

## Understanding the Results

### What is RMSE?
RMSE (Root Mean Squared Error) measures how far predictions are from actual values.

- Target: review_scores_rating (1-5 stars)
- RMSE = 0.41 means predictions are off by ~0.41 stars on average

| RMSE | Quality |
|------|---------|
| 0.0 | Perfect |
| 0.2 | Excellent |
| 0.4 | Decent ← We are here |
| 0.5+ | Needs improvement |

### What is Overfitting?
When Train RMSE << Test RMSE, the model memorized training data but can't generalize.

**Example from Ruth's results:**
- Random Forest (no depth limit): Train=0.16, Test=0.42 → Overfitting!
- Random Forest (depth=5): Train=0.41, Test=0.41 → Good!

---

## Important Rules

1. **Change ONE variable at a time** — This is the whole point
2. **Always include --team_member** — So we know whose run it is
3. **Log everything** — W&B tracks it automatically
4. **Don't delete other people's runs**

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'wandb'"**
```bash
pip install wandb
```

**"wandb: ERROR api_key not configured"**
```bash
wandb login
```

**"FileNotFoundError: data/"**
Make sure you downloaded the data files to the `data/` folder.

---

## Links

- **W&B Project:** https://wandb.ai/ruti-adr-nebius/airbnb-rating-prediction
- **Git Repo:** https://github.com/RuthAdler/airbnb-rating-prediction
