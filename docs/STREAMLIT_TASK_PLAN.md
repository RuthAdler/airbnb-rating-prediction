# Streamlit App - Task Plan

## Goal

We need to build a simple web app where:
1. A user uploads a CSV file with AirBnB listings
2. Our model predicts the rating for each listing
3. The user downloads a CSV file with the predictions

Deadline: Friday 23:59

---

## Model Info

Based on our W&B experiments, we're using:

- Model: XGBoost
- n_estimators: 100
- learning_rate: 0.05
- Scaler: StandardScaler
- Test RMSE: 0.397

This came from the run called `Ella Yakir_xgboost_n100_lr0.05_standard`

---

## Tasks

We have 4 tasks. Please add your name next to the one you'll work on.

| Task | Owner | Status |
|------|-------|--------|
| A: Model Training & Saving | | Not Started |
| B: Inference Preprocessing | | Not Started |
| C: Streamlit App | | Not Started |
| D: Nebius VM Deployment | | Not Started |

Note: Task D should be done last since it depends on A, B, and C being finished.

---

## Task A: Model Training & Saving

### What you're doing
Training our best model and saving it to files so the app can use it.

### Why this matters
Right now our model only exists during training. We need to save it to a file so the Streamlit app can load it and make predictions.

### Steps

1. Create a folder called `models` in the project root

2. Open `run_experiment.py` and find the `run_experiment()` function

3. Add this code at the end of that function (before `wandb.finish()`):

```python
import joblib
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Save the model
joblib.dump(model, 'models/best_model.pkl')
print("Saved: models/best_model.pkl")

# Save the scaler
if scaler is not None:
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved: models/scaler.pkl")

# Save feature columns
joblib.dump(X_train.columns.tolist(), 'models/feature_columns.pkl')
print("Saved: models/feature_columns.pkl")
```

4. Run this command in terminal:
```bash
python run_experiment.py --team_member "Final" --model xgboost --scaler standard --n_estimators 100 --learning_rate 0.05
```

5. Check that these 3 files were created:
```
models/
    best_model.pkl
    scaler.pkl
    feature_columns.pkl
```

6. Push to Git

### How to know you're done
- The 3 files exist in the `models/` folder
- You pushed them to Git
- Other team members can pull and see the files

---

## Task B: Inference Preprocessing

### What you're doing
Creating a new preprocessing function that works on NEW data (data we want predictions for).

### Why this matters
Our current `preprocessing.py` is designed for training. It:
- Expects the rating column to exist (but new data won't have ratings - that's what we're predicting)
- Drops rows with missing data (but we need a prediction for EVERY row)
- Calculates statistics from the data (but we should use the statistics from training)

We need a simpler version for making predictions.

### The key rules for inference preprocessing
1. Never drop any rows - every row needs a prediction
2. Don't expect the rating column to exist
3. Use fixed values for filling missing data (not calculated from the new data)

### Steps

1. Create a new file: `src/preprocessing_inference.py`

2. Use this template and fill in the missing parts based on our existing `preprocessing.py`:

```python
"""
Preprocessing for new data (inference).
This is different from training preprocessing because:
- We don't have the target column (review_scores_rating)
- We can't drop any rows
- We use fixed values instead of calculating from data
"""

import pandas as pd
import numpy as np
import ast

# These are the values we calculated during training
# Use these to fill missing values (don't calculate new ones)
IMPUTATION_VALUES = {
    'host_response_rate_median': 1.0,
    'host_acceptance_rate_median': 0.99,
    'bedrooms_median': 1.0,
    'beds_median': 2.0,
    'bathrooms_mode': 1.0,
}

# These are the exact columns our model expects, in order
FEATURE_COLUMNS = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'description_length_chars', 'description_length_words',
    'distance_to_center', 'estimated_occupancy_l365d',
    'has_host_about', 'has_neighborhood_overview',
    'host_acceptance_rate', 'host_is_superhost',
    'host_response_rate', 'host_response_time_coded',
    'host_tenure_days', 'instant_bookable',
    'log_host_listings_count', 'log_host_total_listings_count',
    'log_price', 'maximum_maximum_nights', 'maximum_minimum_nights',
    'maximum_nights', 'maximum_nights_avg_ntm',
    'minimum_maximum_nights', 'minimum_minimum_nights',
    'minimum_nights', 'minimum_nights_avg_ntm'
]


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess new data for prediction.
    
    Input: Raw AirBnB listings data
    Output: Cleaned data with exactly the columns the model needs
    
    Important: This function should NEVER reduce the number of rows.
    """
    df = df.copy()
    original_row_count = len(df)
    
    # ----- Data Type Conversions -----
    # Copy the conversion code from preprocessing.py
    # (convert dates, percentages, booleans, etc.)
    
    
    # ----- Feature Engineering -----
    # Copy the feature engineering code from preprocessing.py
    # (description_length, has_host_about, log_price, etc.)
    
    
    # ----- Handle Missing Values -----
    # Use the IMPUTATION_VALUES dictionary above
    # Example:
    # df['host_response_rate'] = df['host_response_rate'].fillna(IMPUTATION_VALUES['host_response_rate_median'])
    
    
    # ----- Select Final Features -----
    # Make sure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    X = df[FEATURE_COLUMNS]
    
    # ----- Final Cleanup -----
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)
    
    # Make sure we didn't lose any rows
    assert len(X) == original_row_count, f"Lost rows! Started with {original_row_count}, ended with {len(X)}"
    
    return X
```

3. Test it with some sample data to make sure it works

4. Push to Git

### How to know you're done
- The file `src/preprocessing_inference.py` exists
- It takes raw data and returns a DataFrame with 25 columns
- The number of rows in equals the number of rows out (no drops)
- You pushed it to Git

---

## Task C: Streamlit App

### What you're doing
Building the web interface where users upload files and download predictions.

### Why this matters
This is what the user actually sees and interacts with. It ties together the model (Task A) and preprocessing (Task B).

### Steps

1. Create a file called `app.py` in the project root (not in src/)

2. Use this code:

```python
"""
AirBnB Rating Prediction App
"""

import streamlit as st
import pandas as pd
import joblib
from predictor.preprocessing_inference import preprocess_for_inference

# Page setup
st.set_page_config(page_title="AirBnB Rating Predictor", layout="centered")


# Load the model and scaler (this only runs once)
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler


# Main app
st.title("AirBnB Rating Predictor")
st.write("Upload a CSV file with AirBnB listings to get rating predictions.")

# Try to load the model
try:
    model, scaler = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the file
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df)} rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # Preprocess
    try:
        X = preprocess_for_inference(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # Scale the features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Create output dataframe
    output = pd.DataFrame({'prediction': predictions})

    # Show first few predictions
    st.write("Preview (first 10 rows):")
    st.dataframe(output.head(10))

    # Download button
    csv = output.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

    st.write(f"Done! Generated {len(predictions)} predictions.")
```

3. Test it locally:
```bash
pip install streamlit
streamlit run app.py
```
This will open a browser window at http://localhost:8501

4. Push to Git

### How to know you're done
- The app runs locally without errors
- You can upload a CSV file
- You can download predictions
- The number of predictions matches the number of rows uploaded
- You pushed it to Git

---

## Task D: Nebius VM Deployment

### What you're doing
Setting up a server in the cloud so Franz can access our app from the internet.

### Why this matters
Right now the app only runs on your laptop. We need it running on a server with a public URL so Franz can test it.

### Steps

1. Create a VM on Nebius:
   - OS: Ubuntu 22.04
   - Size: 2 CPU, 4GB RAM is enough
   - Make sure to open port 8501 in the firewall settings

2. Connect to your VM:
```bash
ssh username@YOUR_VM_IP_ADDRESS
```

3. Install what we need:
```bash
sudo apt update
sudo apt install python3-pip git -y
```

4. Get our code:
```bash
git clone https://github.com/RuthAdler/airbnb-rating-prediction.git
cd airbnb-rating-prediction
```

5. Install Python packages:
```bash
pip install -r requirements.txt
pip install streamlit joblib xgboost
```

6. Start the app:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

7. Test it by going to this URL in your browser:
```
http://YOUR_VM_IP_ADDRESS:8501
```

8. Keep it running even after you close the terminal:
```bash
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
```

### How to know you're done
- You can access the app at http://YOUR_VM_IP:8501 from your browser
- You tested uploading a CSV and downloading predictions
- The VM will stay running until Friday 23:59
- You shared the URL with the team

---

## Project Structure When We're Done

```
airbnb-rating-prediction/
    app.py                              <-- Task C
    models/
        best_model.pkl                  <-- Task A
        scaler.pkl                      <-- Task A
        feature_columns.pkl             <-- Task A
    src/
        preprocessing.py                <-- already exists
        preprocessing_inference.py      <-- Task B
    requirements.txt
    STREAMLIT_TASK_PLAN.md
```

---

## Questions?

Post in the group chat.

## Links

- Our W&B experiments: https://wandb.ai/ruti-adr-nebius/airbnb-rating-prediction
- GitHub repo: https://github.com/RuthAdler/airbnb-rating-prediction
- Streamlit documentation: https://docs.streamlit.io/
