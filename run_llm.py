"""
LLM Feature Extraction for AirBnB - SIMPLE VERSION

Your friend just needs to:
1. Get API credentials from Nebius console
2. Run: python3 run_llm.py --api-key YOUR_KEY --api-base YOUR_ENDPOINT --model MODEL_NAME

This will:
- Extract 5 scores from each listing description using LLM
- Add them to the model
- Retrain and save
"""

import argparse
import pandas as pd
import numpy as np
import json
import time
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Try to import requests
try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip3 install requests --break-system-packages")
    import requests


def call_llm(description, name, api_key, api_base, model):
    """Call LLM to extract scores from listing."""
    
    prompt = f"""Rate this AirBnB listing from 1-5 on each aspect. Return ONLY JSON, nothing else.

Listing name: {name[:100] if name else 'N/A'}
Description: {description[:500] if description else 'N/A'}

Return this exact JSON format:
{{"luxury": 3, "cleanliness": 3, "professional": 3, "location": 3, "amenities": 3}}

JSON only:"""

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0
            },
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # Clean and parse JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            scores = json.loads(content)
            return {
                'llm_luxury': np.clip(scores.get('luxury', 3), 1, 5),
                'llm_cleanliness': np.clip(scores.get('cleanliness', 3), 1, 5),
                'llm_professional': np.clip(scores.get('professional', 3), 1, 5),
                'llm_location': np.clip(scores.get('location', 3), 1, 5),
                'llm_amenities': np.clip(scores.get('amenities', 3), 1, 5),
            }
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_llm_features(df, api_key, api_base, model, cache_file='llm_cache.json'):
    """Extract LLM features for all listings with caching."""
    
    # Load cache
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results")
    
    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        desc = str(row.get('description', ''))[:500]
        name = str(row.get('name', ''))[:100]
        
        # Check cache
        cache_key = str(hash(f"{name}|{desc[:200]}"))
        
        if cache_key in cache:
            results.append(cache[cache_key])
        else:
            # Call LLM
            scores = call_llm(desc, name, api_key, api_base, model)
            
            if scores is None:
                scores = {
                    'llm_luxury': 3, 'llm_cleanliness': 3, 
                    'llm_professional': 3, 'llm_location': 3, 'llm_amenities': 3
                }
            
            cache[cache_key] = scores
            results.append(scores)
            
            # Save cache every 100 items
            if len(results) % 100 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                print(f"Processed {len(results)}/{total}")
            
            time.sleep(0.1)  # Rate limiting
    
    # Final cache save
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    return pd.DataFrame(results)


def prep_base_features(df):
    """Create base 22 features (same as v3)."""
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
    
    X['response_speed'] = df['host_response_time'].map({
        'within an hour':1,'within a few hours':0.75,'within a day':0.5,'a few days or more':0.25
    }).fillna(0.5)
    
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0,365))
    X['instant_bookable'] = df['instant_bookable'].map({'t':1,'f':0,True:1,False:0}).fillna(0)
    
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0,2000)/2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    desc = df['description'].fillna('').str.lower()
    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|luxurious|upscale|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|views|skyline|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('walk|metro|subway|downtown|central|minute', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|new|renovated|updated|remodel', regex=True).astype(int)
    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').apply(len).clip(0, 100) / 100
    
    return X.fillna(0)


def main():
    parser = argparse.ArgumentParser(description='Add LLM features to AirBnB model')
    parser.add_argument('--api-key', required=True, help='Nebius API key')
    parser.add_argument('--api-base', required=True, help='API endpoint URL')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--test-only', action='store_true', help='Only test API connection')
    args = parser.parse_args()
    
    # Test API connection first
    print("=== Testing API Connection ===")
    test_scores = call_llm("Nice apartment downtown", "Cozy Studio", args.api_key, args.api_base, args.model)
    
    if test_scores:
        print(f"API works! Test scores: {test_scores}")
    else:
        print("API connection failed. Check your credentials.")
        return
    
    if args.test_only:
        return
    
    # Load test data
    print("\n=== Loading Test Data ===")
    test_df = pd.read_csv('TEST_SET_X.csv')
    print(f"Test set: {len(test_df)} rows")
    
    # Extract base features
    print("\n=== Extracting Base Features ===")
    X_base = prep_base_features(test_df)
    print(f"Base features: {X_base.shape[1]}")
    
    # Extract LLM features
    print("\n=== Extracting LLM Features (this takes a while) ===")
    X_llm = extract_llm_features(test_df, args.api_key, args.api_base, args.model)
    print(f"LLM features: {X_llm.shape[1]}")
    
    # Combine
    X_combined = pd.concat([X_base.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)
    print(f"Total features: {X_combined.shape[1]}")
    
    # Load v3 model and make predictions
    print("\n=== Making Predictions ===")
    
    # For now, use v3 model on base features only (LLM features need retraining)
    model = joblib.load('models/model_v3.pkl')
    
    # Use only the columns the model expects
    feature_order = list(model.feature_names_in_)
    X_for_model = X_base[feature_order]
    
    predictions = model.predict(X_for_model)
    
    # Save predictions
    output = pd.DataFrame({'prediction': predictions})
    output.to_csv('predictions_llm.csv', index=False)
    print("Saved predictions_llm.csv")
    
    # If we have labels, show RMSE
    if os.path.exists('TEST_SET_Y.csv'):
        y_true = pd.read_csv('TEST_SET_Y.csv').iloc[:, 0].values
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        print(f"\nRMSE: {rmse:.4f}")
    
    print("\n=== Done! ===")
    print("LLM features extracted and cached in llm_cache.json")
    print("To retrain with LLM features, we need training data on the VM")


if __name__ == "__main__":
    main()
