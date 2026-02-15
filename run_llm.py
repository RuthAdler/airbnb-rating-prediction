"""
LLM Feature Extraction for AirBnB - FIXED VERSION

Run: python3 run_llm.py --api-key YOUR_KEY --api-base YOUR_ENDPOINT --model MODEL_NAME
"""

import argparse
import pandas as pd
import numpy as np
import json
import time
import os
import joblib
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system("pip3 install requests --break-system-packages")
    import requests


def call_llm(description, name, api_key, api_base, model):
    """Call LLM to extract scores from listing."""
    
    prompt = f"""You are a JSON API. Rate this AirBnB listing 1-5. Output ONLY valid JSON, nothing else.

Name: {name[:100] if name else 'N/A'}
Description: {description[:500] if description else 'N/A'}

Output this JSON with your ratings:
{{"luxury":3,"cleanliness":3,"professional":3,"location":3,"amenities":3}}"""

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
                "max_tokens": 150,
                "temperature": 0
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data['choices'][0]['message']
            
            # Try different response formats
            content = message.get('content')
            if content is None:
                content = message.get('reasoning_content')
            if content is None:
                content = message.get('reasoning')
            if content is None:
                print(f"No content in response: {message}")
                return None
            
            # Clean and parse JSON
            content = str(content).strip()
            
            # Remove markdown code blocks if present
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        content = part
                        break
            
            # Find JSON object in response
            start = content.find("{")
            end = content.find("}") + 1  # Find FIRST closing brace, not last
            if start >= 0 and end > start:
                content = content[start:end]
            else:
                # No JSON found, return default
                return {
                    'llm_luxury': 3, 'llm_cleanliness': 3, 
                    'llm_professional': 3, 'llm_location': 3, 'llm_amenities': 3
                }
            
            scores = json.loads(content)
            return {
                'llm_luxury': np.clip(float(scores.get('luxury', 3)), 1, 5),
                'llm_cleanliness': np.clip(float(scores.get('cleanliness', 3)), 1, 5),
                'llm_professional': np.clip(float(scores.get('professional', 3)), 1, 5),
                'llm_location': np.clip(float(scores.get('location', 3)), 1, 5),
                'llm_amenities': np.clip(float(scores.get('amenities', 3)), 1, 5),
            }
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e} - Content: {content[:200]}")
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
    errors = 0
    
    for idx, (_, row) in enumerate(df.iterrows()):
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
                errors += 1
                scores = {
                    'llm_luxury': 3, 'llm_cleanliness': 3, 
                    'llm_professional': 3, 'llm_location': 3, 'llm_amenities': 3
                }
            
            cache[cache_key] = scores
            results.append(scores)
            
            # Progress update every 50 items
            if (idx + 1) % 50 == 0:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                print(f"Processed {idx + 1}/{total} (errors: {errors})")
            
            time.sleep(0.2)  # Rate limiting
    
    # Final cache save
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    print(f"Done! Total errors: {errors}/{total}")
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
    parser.add_argument('--train', action='store_true', help='Train new model with LLM features')
    args = parser.parse_args()
    
    # Test API connection first
    print("=== Testing API Connection ===")
    test_scores = call_llm("Lovely 2BR apartment in downtown. Very clean and modern with great city views.", 
                           "Cozy Modern Downtown Apartment", 
                           args.api_key, args.api_base, args.model)
    
    if test_scores:
        print(f"✓ API works! Test scores: {test_scores}")
    else:
        print("✗ API connection failed. Check your credentials.")
        return
    
    if args.test_only:
        return
    
    # Check if training data exists
    data_dir = Path('data')
    has_training_data = data_dir.exists() and list(data_dir.glob('listings*.csv'))
    
    if args.train and has_training_data:
        print("\n=== Training Mode ===")
        
        # Load training data
        print("Loading training data...")
        datasets = []
        for f in data_dir.glob('listings*.csv'):
            if 'TEST' not in f.name.upper():
                print(f"  Loading {f.name}")
                datasets.append(pd.read_csv(f))
        
        train_df = pd.concat(datasets, ignore_index=True)
        train_df = train_df.dropna(subset=['review_scores_rating'])
        y_train = train_df['review_scores_rating'].values
        print(f"Training data: {len(train_df)} rows")
        
        # Extract features
        print("\nExtracting base features...")
        X_base = prep_base_features(train_df)
        
        print("\nExtracting LLM features for training data...")
        X_llm = extract_llm_features(train_df, args.api_key, args.api_base, args.model, 
                                     cache_file='llm_cache_train.json')
        
        # Combine features
        X_train = pd.concat([X_base.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)
        print(f"\nTotal features: {X_train.shape[1]}")
        
        # Train model
        print("\nTraining model...")
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        print(f"CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, 'models/model_llm.pkl')
        joblib.dump(list(X_train.columns), 'models/feature_columns_llm.pkl')
        print("\nSaved: models/model_llm.pkl")
        
    else:
        # Just process test data
        print("\n=== Processing Test Data ===")
        
        if not os.path.exists('TEST_SET_X.csv'):
            print("TEST_SET_X.csv not found!")
            return
        
        test_df = pd.read_csv('TEST_SET_X.csv')
        print(f"Test set: {len(test_df)} rows")
        
        # Extract features
        print("\nExtracting base features...")
        X_base = prep_base_features(test_df)
        
        print("\nExtracting LLM features...")
        X_llm = extract_llm_features(test_df, args.api_key, args.api_base, args.model)
        
        # Save LLM features
        X_llm.to_csv('llm_features_test.csv', index=False)
        print("\nSaved: llm_features_test.csv")
        
        # Combine and predict with v3 model
        print("\nMaking predictions with v3 model + LLM features info...")
        model = joblib.load('models/model_v3.pkl')
        feature_order = list(model.feature_names_in_)
        X_for_model = X_base[feature_order]
        predictions = model.predict(X_for_model)
        
        # Save predictions
        output = pd.DataFrame({'prediction': predictions})
        output.to_csv('predictions_with_llm.csv', index=False)
        print("Saved: predictions_with_llm.csv")
        
        # Calculate RMSE if labels available
        if os.path.exists('TEST_SET_Y.csv'):
            from sklearn.metrics import mean_squared_error
            y_true = pd.read_csv('TEST_SET_Y.csv').iloc[:, 0].values
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            print(f"\nRMSE: {rmse:.4f}")
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
