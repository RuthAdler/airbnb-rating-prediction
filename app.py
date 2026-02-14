"""
AirBnB Rating Prediction App

Supports:
- Model v2 (simple, 15 features)
- Model v3 (with text keywords, 22 features) - GenAI submission
"""

import joblib
import os
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")

# Model paths
MODELS = {
    "v2 - Simple (15 features)": {
        "model": "models/best_model.pkl",
        "preprocess": "v2"
    },
    "v3 - GenAI (22 features)": {
        "model": "models/model_v3.pkl",
        "preprocess": "v3"
    }
}


def preprocess_v2(df):
    """Simple preprocessing - 15 features."""
    import numpy as np
    
    X = pd.DataFrame()
    X['accommodates'] = df['accommodates'].fillna(2)
    X['bathrooms'] = df['bathrooms'].fillna(1)
    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)
    
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100
    
    X['is_superhost'] = df['host_is_superhost'].map({'t':1,'f':0,True:1,False:0}).fillna(0)
    
    df2 = df.copy()
    df2['host_since'] = pd.to_datetime(df2['host_since'], errors='coerce', dayfirst=True)
    df2['last_scraped'] = pd.to_datetime(df2['last_scraped'], errors='coerce', dayfirst=True)
    X['host_days_log'] = np.log1p((df2['last_scraped']-df2['host_since']).dt.days.fillna(0).clip(lower=0))
    
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0,365))
    X['instant_bookable'] = df['instant_bookable'].map({'t':1,'f':0,True:1,False:0}).fillna(0)
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0,2000)/2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    X['response_speed'] = df['host_response_time'].map({
        'within an hour':1,'within a few hours':0.75,'within a day':0.5,'a few days or more':0.25
    }).fillna(0.5)
    
    return X.fillna(0)


def preprocess_v3(df):
    """GenAI preprocessing - 22 features (15 + 7 text keywords)."""
    import numpy as np
    
    # Start with v2 features
    X = preprocess_v2(df)
    
    # Add text keyword features
    desc = df['description'].fillna('').str.lower()
    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|luxurious|upscale|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|views|skyline|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('walk|metro|subway|downtown|central|minute', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|new|renovated|updated|remodel', regex=True).astype(int)
    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').apply(len).clip(0, 100) / 100
    
    return X.fillna(0)


# Main app
st.title("🏠 AirBnB Rating Predictor")
st.write("Upload a CSV file with AirBnB listings to get rating predictions.")

# Model selection
st.sidebar.header("Settings")
available_models = []
for name, config in MODELS.items():
    if os.path.exists(config["model"]):
        available_models.append(name)

if not available_models:
    st.error("No models found! Run train_simple.py or train_v3.py first.")
    st.stop()

selected_model = st.sidebar.radio("Select Model", available_models)
model_config = MODELS[selected_model]

# Load model
try:
    model = joblib.load(model_config["model"])
    st.success(f"✓ Loaded {selected_model}")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"📁 Uploaded **{len(df)}** rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    
    # Show preview
    with st.expander("Preview data"):
        st.dataframe(df.head())
    
    if st.button("🚀 Generate Predictions", type="primary"):
        try:
            # Preprocess
            if model_config["preprocess"] == "v2":
                X = preprocess_v2(df)
            else:
                X = preprocess_v3(df)
            
            st.write(f"Features: {len(X.columns)}")
            
            # Predict
            predictions = model.predict(X)
            
            # Results
            st.success(f"✅ Generated **{len(predictions)}** predictions")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{predictions.mean():.3f}")
            with col2:
                st.metric("Min", f"{predictions.min():.3f}")
            with col3:
                st.metric("Max", f"{predictions.max():.3f}")
            
            # Output
            output = pd.DataFrame({'prediction': predictions})
            
            st.write("**Preview:**")
            st.dataframe(output.head(10))
            
            # Download
            csv = output.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
