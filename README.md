cat > README.md << 'EOF'
# AirBnB Listing Rating Prediction

> DS in Production 2025-6, Nebius Academy IL

---

## Overview

This project predicts the average overall rating (`review_scores_rating`) of AirBnB listings. Trained on NYC and LA data, with the goal of generalizing to any city.

---

## Purpose

This project is part of the DS in Production course. The goal is to practice:
- Structuring a Python project with proper folder organization
- Separating code into reusable modules
- Using Git workflow (branches, pull requests, code reviews)
- Building code that generalizes to unseen data

The instructor (`nebius-franz`) has been added as a collaborator and will test the project.

---

## Project Structure
```
airbnb-rating-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # Data files (not tracked in git)
│   ├── listings LA.csv
│   └── listings NYC.csv
├── notebooks/               # Jupyter notebooks
└── src/                     # Source code modules
    ├── __init__.py
    ├── data_loading.py      # Data loading utilities
    ├── preprocessing.py     # Data cleaning
    ├── geo_processing.py    # Geographic features
    └── visualization.py     # Plotting functions
```

---

## Installation
```bash
git clone https://github.com/RuthAdler/airbnb-rating-prediction.git
cd airbnb-rating-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## Data

Download the data files from [Google Drive](https://drive.google.com/drive/u/0/folders/1d-IlNVY2rgYmBh6G_4YDg37OhvPGceWK) and place them in the `data/` folder.

---

## Usage
```python
from src.data_loading import load_all_listings, validate_columns_match

datasets = load_all_listings("data")
print(validate_columns_match(datasets))  # True
```

---

## Team

- Ruth Adler
- Ido Friedmann
- Ella Yakir
- Rosemary Lavender

EOF