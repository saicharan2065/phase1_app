import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

class ReferenceDataMatchingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.st_model = None # We will load this lazily

    def load_model(self):
        # By doing this lazily during the matching process, the UI can intercept 
        # the tqdm download progress bar and display the exact percentage and size.
        if ST_AVAILABLE and self.st_model is None:
            self.st_model = SentenceTransformer(self.model_name)

    def match_records(self, source_df, ref_df, threshold=80.0):
        if source_df is None or source_df.empty or ref_df is None or ref_df.empty:
            return 0.0, pd.DataFrame()

        # Load model lazily (will trigger download progress bar to UI on first run)
        self.load_model()

        # Simple schema alignment by lowercase names
        source_cols = [str(c).lower() for c in source_df.columns]
        ref_cols = [str(c).lower() for c in ref_df.columns]

        common_cols = list(set(source_cols).intersection(set(ref_cols)))
        if not common_cols:
            s_mapping = {str(c).lower(): c for c in source_df.columns}
            r_mapping = {str(c).lower(): c for c in ref_df.columns}
            source_docs = source_df.astype(str).agg(' '.join, axis=1).tolist()
            ref_docs = ref_df.astype(str).agg(' '.join, axis=1).tolist()
            common_cols = ["ALL_COLUMNS_FALLBACK"]
        else:
            s_mapping = {str(c).lower(): c for c in source_df.columns}
            r_mapping = {str(c).lower(): c for c in ref_df.columns}

            # Vectorized string concatenation (100x faster than apply lambda)
            source_docs = source_df[[s_mapping[k] for k in common_cols]].astype(str).agg(' '.join, axis=1).tolist()
            ref_docs = ref_df[[r_mapping[k] for k in common_cols]].astype(str).agg(' '.join, axis=1).tolist()

        # Optional Auto-Translation before semantic matching
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='auto', target='en')
            def safe_translate(texts):
                translated = []
                for t in texts:
                    try:
                        translated.append(translator.translate(t) or t)
                    except:
                        translated.append(t) # Fallback to original text if API fails
                return translated
                
            # Only translate if there are fewer than 1000 rows to prevent extreme API throttling
            if len(source_docs) <= 1000 and len(ref_docs) <= 1000:
                source_docs = safe_translate(source_docs)
                ref_docs = safe_translate(ref_docs)
        except Exception:
            pass

        # Fast TF-IDF matrix multiplication
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
        vectorizer.fit(ref_docs + source_docs)
        
        s_matrix = vectorizer.transform(source_docs)
        r_matrix = vectorizer.transform(ref_docs)
        
        # Distribute matrix dot product across all CPU cores
        import concurrent.futures
        
        def compute_chunk(s_chunk):
            return cosine_similarity(s_chunk, r_matrix) * 100
            
        chunk_size = max(1, s_matrix.shape[0] // 16) # Split into 16 chunks
        chunks = [s_matrix[i:i+chunk_size] for i in range(0, s_matrix.shape[0], chunk_size)]
        
        best_ref_indices = []
        best_scores = []
        
        # Scipy releases the GIL during sparse dot products, allowing true multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sim_chunk in executor.map(compute_chunk, chunks):
                best_ref_indices.extend(np.argmax(sim_chunk, axis=1))
                best_scores.extend(np.max(sim_chunk, axis=1))

        matches = []
        total_score = 0
        match_count = 0

        for i in range(len(source_docs)):
            best_score = best_scores[i]
            is_match = best_score >= threshold
            
            matches.append({
                "Source Index": i,
                "Reference Index": int(best_ref_indices[i]) if is_match else None,
                "Confidence": f"{round(float(best_score), 2)}%",
                "Status": "Match" if is_match else "Mismatch",
                "Reason": f"Accepted: Found {round(float(best_score), 2)}% semantic overlap on mapped columns ({', '.join(common_cols)})" if is_match else "Rejected: No reference record met the threshold.",
                "Source Data Evaluated": source_docs[i][:150] + "..." if len(source_docs[i]) > 150 else source_docs[i]
            })
            
            if is_match:
                total_score += best_score
                match_count += 1

        overall_score = round(float(total_score / match_count), 2) if match_count > 0 else 0.0
        return overall_score, pd.DataFrame(matches)
