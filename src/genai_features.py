"""
GenAI Feature Extraction for AirBnB Rating Prediction.

This module adds:
1. Text embeddings (reduced via PCA)
2. LLM-extracted scores (luxury, cleanliness, etc.)

Usage:
    # During training:
    extractor = GenAIFeatureExtractor()
    genai_features = extractor.fit_transform(train_df)
    extractor.save("models/genai_extractor.pkl")
    
    # During inference:
    extractor = GenAIFeatureExtractor.load("models/genai_extractor.pkl")
    genai_features = extractor.transform(test_df)
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Optional
import time


class EmbeddingExtractor:
    """Extract and reduce text embeddings using sentence-transformers + PCA."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_components: int = 15):
        self.model_name = model_name
        self.n_components = n_components
        self.model = None
        self.pca = None
        self.is_fitted = False
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed.\n"
                    "Run: pip install sentence-transformers"
                )
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get raw embeddings for texts."""
        self._load_model()
        
        # Clean texts
        clean_texts = [str(t) if pd.notna(t) else "" for t in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            clean_texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def fit(self, df: pd.DataFrame, text_column: str = 'description'):
        """Fit PCA on training data embeddings."""
        from sklearn.decomposition import PCA
        
        print(f"Fitting embeddings on '{text_column}'...")
        
        # Get embeddings
        texts = df[text_column].tolist() if text_column in df.columns else [""] * len(df)
        embeddings = self._get_embeddings(texts)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(embeddings)
        
        explained_var = sum(self.pca.explained_variance_ratio_) * 100
        print(f"PCA fitted: {self.n_components} components explain {explained_var:.1f}% variance")
        
        self.text_column = text_column
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform text to reduced embeddings."""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        # Get embeddings
        texts = df[self.text_column].tolist() if self.text_column in df.columns else [""] * len(df)
        embeddings = self._get_embeddings(texts)
        
        # Apply PCA
        reduced = self.pca.transform(embeddings)
        
        # Create DataFrame
        columns = [f'emb_{i}' for i in range(self.n_components)]
        result = pd.DataFrame(reduced, columns=columns, index=df.index)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, text_column: str = 'description') -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, text_column)
        return self.transform(df)


class LLMFeatureExtractor:
    """Extract structured features using LLM API."""
    
    # The prompt template for extracting features
    PROMPT_TEMPLATE = """Analyze this AirBnB listing and rate each aspect from 1-5.
Return ONLY a JSON object with these exact keys, nothing else.

Listing description:
{description}

Listing name:
{name}

Amenities:
{amenities}

Return JSON with these keys (all values 1-5):
- luxury_score: How luxurious/upscale does this listing seem?
- cleanliness_emphasis: How much does the listing emphasize cleanliness?
- host_professionalism: How professional does the host seem?
- location_appeal: How appealing is the location based on description?
- amenities_quality: Overall quality of amenities offered?

JSON only:"""

    def __init__(
        self, 
        api_type: str = "openai",  # "openai" or "anthropic"
        api_key: str = None,
        api_base: str = None,  # For Nebius or other OpenAI-compatible APIs
        model: str = None,
        cache_path: str = None
    ):
        self.api_type = api_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.model = model or self._default_model()
        self.cache_path = cache_path
        self.cache = {}
        self.is_fitted = True  # No fitting needed for LLM
        
        # Load cache if exists
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} cached LLM responses")
    
    def _default_model(self):
        if self.api_type == "anthropic":
            return "claude-3-haiku-20240307"
        else:
            return "gpt-3.5-turbo"
    
    def _get_cache_key(self, description: str, name: str) -> str:
        """Create a cache key from inputs."""
        text = f"{name}|||{description[:200]}"  # First 200 chars
        return str(hash(text))
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API."""
        try:
            import openai
            
            client_kwargs = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            
            client = openai.OpenAI(**client_kwargs)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
    
    def _extract_single(self, description: str, name: str, amenities: str) -> Dict[str, float]:
        """Extract features for a single listing."""
        
        # Check cache
        cache_key = self._get_cache_key(description, name)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare prompt
        prompt = self.PROMPT_TEMPLATE.format(
            description=str(description)[:500] if pd.notna(description) else "No description",
            name=str(name) if pd.notna(name) else "No name",
            amenities=str(amenities)[:300] if pd.notna(amenities) else "Not listed"
        )
        
        # Call API
        try:
            if self.api_type == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                response = self._call_openai(prompt)
            
            # Parse JSON from response
            # Handle cases where LLM adds markdown
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response)
            
            # Validate and clip values
            features = {
                'llm_luxury_score': np.clip(float(result.get('luxury_score', 3)), 1, 5),
                'llm_cleanliness_emphasis': np.clip(float(result.get('cleanliness_emphasis', 3)), 1, 5),
                'llm_host_professionalism': np.clip(float(result.get('host_professionalism', 3)), 1, 5),
                'llm_location_appeal': np.clip(float(result.get('location_appeal', 3)), 1, 5),
                'llm_amenities_quality': np.clip(float(result.get('amenities_quality', 3)), 1, 5),
            }
            
            # Cache result
            self.cache[cache_key] = features
            
            return features
            
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            # Return neutral scores on failure
            return {
                'llm_luxury_score': 3.0,
                'llm_cleanliness_emphasis': 3.0,
                'llm_host_professionalism': 3.0,
                'llm_location_appeal': 3.0,
                'llm_amenities_quality': 3.0,
            }
    
    def transform(self, df: pd.DataFrame, batch_delay: float = 0.1) -> pd.DataFrame:
        """Extract LLM features for all rows."""
        
        results = []
        total = len(df)
        
        print(f"Extracting LLM features for {total} listings...")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            description = row.get('description', '')
            name = row.get('name', '')
            amenities = row.get('amenities', '')
            
            features = self._extract_single(description, name, amenities)
            results.append(features)
            
            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total}")
                # Save cache periodically
                if self.cache_path:
                    self._save_cache()
            
            # Rate limiting
            time.sleep(batch_delay)
        
        # Save final cache
        if self.cache_path:
            self._save_cache()
        
        return pd.DataFrame(results, index=df.index)
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.cache_path:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f)


class GenAIFeatureExtractor:
    """
    Combined GenAI feature extractor.
    
    Combines:
    - Text embeddings (reduced via PCA)
    - LLM-extracted scores
    """
    
    def __init__(
        self,
        use_embeddings: bool = True,
        use_llm: bool = True,
        embedding_components: int = 15,
        llm_api_type: str = "openai",
        llm_api_key: str = None,
        llm_api_base: str = None,
        llm_model: str = None,
        cache_dir: str = "models/cache"
    ):
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm
        self.embedding_components = embedding_components
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize extractors
        if use_embeddings:
            self.embedding_extractor = EmbeddingExtractor(n_components=embedding_components)
        else:
            self.embedding_extractor = None
            
        if use_llm:
            self.llm_extractor = LLMFeatureExtractor(
                api_type=llm_api_type,
                api_key=llm_api_key,
                api_base=llm_api_base,
                model=llm_model,
                cache_path=os.path.join(cache_dir, "llm_cache.json")
            )
        else:
            self.llm_extractor = None
        
        self.is_fitted = False
        self.feature_columns = []
    
    def fit(self, df: pd.DataFrame) -> 'GenAIFeatureExtractor':
        """Fit the extractors on training data."""
        
        if self.use_embeddings:
            self.embedding_extractor.fit(df, text_column='description')
        
        # LLM doesn't need fitting
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data to GenAI features."""
        
        if not self.is_fitted and self.use_embeddings:
            raise ValueError("Must fit before transform (for embeddings PCA)")
        
        features_list = []
        
        # Embeddings
        if self.use_embeddings and self.embedding_extractor:
            print("Extracting embeddings...")
            emb_features = self.embedding_extractor.transform(df)
            features_list.append(emb_features)
            print(f"  Added {len(emb_features.columns)} embedding features")
        
        # LLM features
        if self.use_llm and self.llm_extractor:
            print("Extracting LLM features...")
            llm_features = self.llm_extractor.transform(df)
            features_list.append(llm_features)
            print(f"  Added {len(llm_features.columns)} LLM features")
        
        # Combine
        if features_list:
            result = pd.concat(features_list, axis=1)
        else:
            result = pd.DataFrame(index=df.index)
        
        self.feature_columns = list(result.columns)
        print(f"Total GenAI features: {len(self.feature_columns)}")
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def save(self, path: str):
        """Save the extractor (PCA model, config, cache)."""
        save_dict = {
            'use_embeddings': self.use_embeddings,
            'use_llm': self.use_llm,
            'embedding_components': self.embedding_components,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
        }
        
        # Save embedding PCA
        if self.use_embeddings and self.embedding_extractor:
            save_dict['embedding_pca'] = self.embedding_extractor.pca
            save_dict['embedding_text_column'] = self.embedding_extractor.text_column
            save_dict['embedding_model_name'] = self.embedding_extractor.model_name
        
        joblib.dump(save_dict, path)
        print(f"GenAI extractor saved to {path}")
    
    @classmethod
    def load(cls, path: str, llm_api_key: str = None, llm_api_base: str = None) -> 'GenAIFeatureExtractor':
        """Load the extractor."""
        save_dict = joblib.load(path)
        
        extractor = cls(
            use_embeddings=save_dict['use_embeddings'],
            use_llm=save_dict['use_llm'],
            embedding_components=save_dict['embedding_components'],
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base
        )
        
        extractor.feature_columns = save_dict['feature_columns']
        extractor.is_fitted = save_dict['is_fitted']
        
        # Restore embedding PCA
        if extractor.use_embeddings and 'embedding_pca' in save_dict:
            extractor.embedding_extractor.pca = save_dict['embedding_pca']
            extractor.embedding_extractor.text_column = save_dict['embedding_text_column']
            extractor.embedding_extractor.model_name = save_dict['embedding_model_name']
            extractor.embedding_extractor.is_fitted = True
        
        print(f"GenAI extractor loaded from {path}")
        return extractor


# Feature columns that GenAI adds (for reference)
GENAI_FEATURE_COLUMNS = [
    # Embeddings (15)
    'emb_0', 'emb_1', 'emb_2', 'emb_3', 'emb_4',
    'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9',
    'emb_10', 'emb_11', 'emb_12', 'emb_13', 'emb_14',
    # LLM scores (5)
    'llm_luxury_score',
    'llm_cleanliness_emphasis', 
    'llm_host_professionalism',
    'llm_location_appeal',
    'llm_amenities_quality',
]
