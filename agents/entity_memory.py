import time
import pandas as pd
import numpy as np
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psutil

class EntityMemoryIndex:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EntityMemoryIndex, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
            
    def _initialize(self):
        self.is_built = False
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100000)
        self.tfidf_matrix = None
        self.entity_data = pd.DataFrame()
        self.build_lock = threading.Lock()
        self.status = "IDLE"
        
    def build_index(self, dataset_id, max_records=50000):
        with self.build_lock:
            self.status = f"PULLING REAL DATA: Fetching {max_records} records from {dataset_id}..."
            try:
                from data.dataset_manager import DatasetManager
                dm = DatasetManager()
                
                if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                    raise RuntimeError("No valid target_dataset provided.")
                    
                ds = dm._load_dataset_records_sync(dataset_id, str(max_records))
                if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                    raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
                
                self.status = "EXTRACTING: Converting REAL rows to text blobs..."
                
                if isinstance(ds, pd.DataFrame):
                    df = ds.copy()
                else:
                    df = pd.DataFrame([ds[i] for i in range(len(ds))])
                    
                # Create a rich text representation of every row
                text_blobs = []
                for _, row in df.iterrows():
                    blob = " ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                    text_blobs.append(blob)
                    
                df['vector_text'] = text_blobs
                self.entity_data = df
                
                self.status = "EMBEDDING: Generating massive TF-IDF matrices in System RAM..."
                self.tfidf_matrix = self.vectorizer.fit_transform(self.entity_data['vector_text'])
                
                self.is_built = True
                ram_used = psutil.Process().memory_info().rss / (1024 * 1024)
                self.status = f"READY: Indexed {len(df)} real entities. Process RAM footprint: {ram_used:.1f} MB."
            except Exception as e:
                self.status = f"CRASH: {str(e)}"
                self.is_built = False
                
    def search(self, query, top_k=10):
        if not self.is_built or self.tfidf_matrix is None:
            return pd.DataFrame({"Error": ["Index not built. Please build index first."]})
            
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top K indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score > 0.01: # Filter complete misses
                    row = self.entity_data.iloc[idx].copy()
                    # Determine a mock risk score based on generic heuristic for demo if not present
                    risk = row.get("risk_score", row.get("Amount", np.random.randint(10, 99)))
                    
                    # Try to find an entity ID column
                    entity_id = row.get("Account", row.get("nameDest", row.get("nameOrig", f"ENT-{idx}")))
                    
                    results.append({
                        "Entity ID": entity_id,
                        "Similarity Score": f"{score:.3f}",
                        "Assigned Risk": risk,
                        "Underlying Real Data": row['vector_text'][:150] + "..."
                    })
                    
            if not results:
                return pd.DataFrame({"Result": ["No significant semantic matches found."]})
                
            return pd.DataFrame(results)
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
            
    def clear(self):
        with self.build_lock:
            self.tfidf_matrix = None
            self.entity_data = pd.DataFrame()
            self.is_built = False
            self.status = "CLEARED: Memory released."

entity_memory_index = EntityMemoryIndex()
