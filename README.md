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
│
├── data/
│   ├── listings LA.csv
│   └── listings NYC.csv
│
├── notebooks/
│   ├── airbnb_baseline.ipynb
│   └── PClass1 AirBnB EDA.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── geo_processing.py
│   ├── train.py
│   ├── predict.py
│   ├── results.py
│   └── visualization.py
│
├── main.py
├── README.md
└── requirements.txt

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

## Contributing

### 1. Clone the repo (first time only)
```bash
git clone https://github.com/RuthAdler/airbnb-rating-prediction.git
cd airbnb-rating-prediction
```

### 2. Create your branch
```bash
git checkout -b feature/your-feature-name
```
Examples: `feature/geo-processing`, `feature/preprocessing`, `feature/visualization`

### 3. Do your work
Edit your file in `src/`

### 4. Commit and push
```bash
git add .
git commit -m "Your message here"
git push -u origin feature/your-feature-name
```

### 5. Open a Pull Request
Go to GitHub and click "Compare & pull request". Ask a teammate to review.

---

## Team

- Ruth Adler
- Ido Friedmann
- Ella Yakir
- Rosemary Lavender

EOF
