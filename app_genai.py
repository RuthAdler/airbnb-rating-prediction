"""
AirBnB Rating Prediction App - GenAI Version

This app supports both:
- Original model (25 features)
- GenAI model (45 features with embeddings + LLM scores)
"""

import joblib
import os
import streamlit as st
import pandas as pd

# Config - adjust these paths as needed
MODEL_PATH_ORIGINAL = "models/best_model.pkl"
MODEL_PATH_GENAI = "models/model_genai.pkl"
SCALER_PATH_GENAI = "models/scaler_genai.pkl"
GENAI_EXTRACTOR_PATH = "models/genai_extractor.pkl"

# Page setup
st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")


@st.cache_resource
def load_original_model():
    """Load the original (non-GenAI) model."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_PATH_ORIGINAL)
    return joblib.load(model_path)


@st.cache_resource
def load_genai_resources():
    """Load GenAI model and related resources."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, MODEL_PATH_GENAI)
    scaler_path = os.path.join(current_dir, SCALER_PATH_GENAI)
    extractor_path = os.path.join(current_dir, GENAI_EXTRACTOR_PATH)
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # Load GenAI extractor
    from src.genai_features import GenAIFeatureExtractor
    extractor = GenAIFeatureExtractor.load(extractor_path)
    
    return model, scaler, extractor


def check_available_models():
    """Check which models are available."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    has_original = os.path.exists(os.path.join(current_dir, MODEL_PATH_ORIGINAL))
    has_genai = os.path.exists(os.path.join(current_dir, MODEL_PATH_GENAI))
    
    return has_original, has_genai


# Main app
st.title("🏠 AirBnB Rating Predictor")
st.write("Upload a CSV file with AirBnB listings to get rating predictions.")

# Check available models
has_original, has_genai = check_available_models()

if not has_original and not has_genai:
    st.error("No models found! Please train a model first.")
    st.stop()

# Model selection
st.sidebar.header("Settings")

model_options = []
if has_original:
    model_options.append("Original (25 features)")
if has_genai:
    model_options.append("GenAI (45 features)")

selected_model = st.sidebar.radio("Select Model", model_options, index=len(model_options)-1)

# LLM settings (only for GenAI model)
use_llm = False
llm_api_key = None

if "GenAI" in selected_model:
    st.sidebar.subheader("GenAI Options")
    use_llm = st.sidebar.checkbox("Use LLM features", value=False, 
                                   help="Requires API key. Slower but potentially more accurate.")
    
    if use_llm:
        llm_api_key = st.sidebar.text_input("API Key", type="password",
                                             help="OpenAI or Anthropic API key")
        if not llm_api_key:
            st.sidebar.warning("Enter API key to use LLM features")
            use_llm = False

# Load selected model
try:
    if "Original" in selected_model:
        model = load_original_model()
        st.success("✓ Original model loaded")
        use_genai = False
    else:
        model, scaler, genai_extractor = load_genai_resources()
        st.success("✓ GenAI model loaded")
        use_genai = True
        
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read file
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"📁 Uploaded **{len(df)}** rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    
    # Show preview
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head())
    
    # Process button
    if st.button("🚀 Generate Predictions", type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if use_genai:
                # GenAI preprocessing
                status_text.text("Step 1/3: Original preprocessing...")
                progress_bar.progress(10)
                
                from predictor.preprocessing_inference import preprocess_for_inference
                X_original = preprocess_for_inference(df)
                
                status_text.text("Step 2/3: Extracting GenAI features...")
                progress_bar.progress(30)
                
                # Configure LLM if enabled
                if use_llm and llm_api_key:
                    genai_extractor.use_llm = True
                    genai_extractor.llm_extractor.api_key = llm_api_key
                else:
                    genai_extractor.use_llm = False
                
                X_genai = genai_extractor.transform(df)
                
                # Combine features
                X_original = X_original.reset_index(drop=True)
                X_genai = X_genai.reset_index(drop=True)
                X = pd.concat([X_original, X_genai], axis=1)
                
                # Scale
                if scaler is not None:
                    X = scaler.transform(X)
                
            else:
                # Original preprocessing
                status_text.text("Preprocessing...")
                progress_bar.progress(30)
                
                from predictor.preprocessing_inference import preprocess_for_inference
                X = preprocess_for_inference(df)
                X = X.values
            
            # Predict
            status_text.text("Step 3/3: Generating predictions...")
            progress_bar.progress(70)
            
            predictions = model.predict(X)
            
            progress_bar.progress(100)
            status_text.text("Done!")
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.exception(e)
            st.stop()
        
        # Create output
        output = pd.DataFrame({'prediction': predictions})
        
        # Show results
        st.success(f"✅ Generated **{len(predictions)}** predictions")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Rating", f"{predictions.mean():.2f}")
        with col2:
            st.metric("Min Rating", f"{predictions.min():.2f}")
        with col3:
            st.metric("Max Rating", f"{predictions.max():.2f}")
        
        # Preview
        st.write("**Preview (first 10 rows):**")
        st.dataframe(output.head(10))
        
        # Distribution chart
        st.write("**Prediction Distribution:**")
        st.bar_chart(pd.cut(predictions, bins=10).value_counts().sort_index())
        
        # Download button
        csv = output.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
if "GenAI" in selected_model:
    st.sidebar.markdown("- 25 original features")
    st.sidebar.markdown("- 15 embedding features")
    st.sidebar.markdown("- 5 LLM features (if enabled)")
else:
    st.sidebar.markdown("- 25 original features")
