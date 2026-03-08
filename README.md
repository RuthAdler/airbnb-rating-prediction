# AirBnB Listing Rating Prediction

Predicting `review_scores_rating` for AirBnB listings using listing, host, and text features — designed to generalize across cities.

---

## Results

| Evaluation Set | Baseline RMSE | Model RMSE | Improvement |
|----------------|--------------|------------|-------------|
| Chicago (validation) | 0.4058 | 0.378 | -6.8% |
| Sydney (test) | 0.4412 | 0.4122 | -6.6% |

**Model:** XGBoost (n_estimators=300, max_depth=4, learning_rate=0.05)
**Training data:** Los Angeles (45,886 listings) + New York City (36,111 listings)
**Challenge:** 72% of ratings fall between 4.5–5.0 (median: 4.82) — strong ceiling effect

> Full analysis: [docs/report.pdf](docs/report.pdf) | [docs/presentation.pdf](docs/presentation.pdf)

---

## Features

22 city-agnostic features, grouped by category:

| Category | Features |
|----------|----------|
| **Property** | accommodates, bathrooms, bedrooms, beds, room_ratio |
| **Host** | host_response_rate, host_acceptance_rate, is_superhost, host_days_log, response_speed |
| **Booking** | minimum_nights, instant_bookable |
| **Text metadata** | has_description, desc_length, has_host_about, has_neighborhood, name_length |
| **Keywords** | mentions_clean, mentions_luxury, mentions_view, mentions_location, mentions_modern |

Price and geographic features were intentionally excluded — they hurt generalization to unseen cities.

---

## Project Structure

```
airbnb-rating-prediction/
├── data/                    # Not included (download separately)
├── models/                  # Trained model artifacts (.pkl)
├── notebooks/               # EDA and baseline experiments
├── docs/                    # Report and presentation
├── src/
│   ├── data_loading.py      # Load and validate CSV datasets
│   ├── features.py          # Feature engineering pipeline
│   ├── feature_sets.py      # Feature ablation configurations
│   ├── train.py             # Model training script
│   └── baseline.py          # Dummy baseline model
├── app.py                   # Streamlit inference app
├── run_experiment.py        # W&B experiment tracking
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/RuthAdler/airbnb-rating-prediction.git
cd airbnb-rating-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

Download listings CSVs from [Google Drive](https://drive.google.com/drive/u/0/folders/1d-IlNVY2rgYmBh6G_4YDg37OhvPGceWK) and place them in `data/`.

---

## Usage

### Train a model

```bash
python -m src.train --data-dir data --output-dir models
```

Evaluates Ridge, GradientBoosting, and XGBoost via 5-fold CV and saves the best model.

### Run the Streamlit app

```bash
streamlit run app.py
```

Upload a listings CSV to get predictions. Optionally upload true labels to evaluate RMSE against the dummy baseline.

### Run experiments with W&B tracking

```bash
python run_experiment.py --model xgboost --feature-set v0 --data-dir data
```

---

## Team

Ruth Adler · Ido Friedmann · Ella Yakir · Rosemary Lavender
