import pandas as pd
from rapidfuzz import fuzz
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

class ReferenceDataMatchingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.st_model = SentenceTransformer(model_name) if ST_AVAILABLE else None

    def match_records(self, source_df, ref_df, threshold=80.0):
        if source_df is None or source_df.empty or ref_df is None or ref_df.empty:
            return 0.0, pd.DataFrame()

        # Simple schema alignment by lowercase names
        source_cols = [str(c).lower() for c in source_df.columns]
        ref_cols = [str(c).lower() for c in ref_df.columns]

        # Use common columns for string matching
        common_cols = list(set(source_cols).intersection(set(ref_cols)))
        if not common_cols:
            return 0.0, pd.DataFrame([{"Error": "No common schema alignment found"}])

        source_records = source_df.to_dict('records')
        ref_records = ref_df.to_dict('records')

        matches = []
        total_score = 0
        match_count = 0

        # Mapping original column names
        s_mapping = {str(c).lower(): c for c in source_df.columns}
        r_mapping = {str(c).lower(): c for c in ref_df.columns}

        for i, s_rec in enumerate(source_records):
            best_score = 0
            best_ref = None
            
            # Create a string representation for fuzzy matching
            s_str = " ".join([str(s_rec[s_mapping[k]]) for k in common_cols if pd.notna(s_rec[s_mapping[k]])])
            
            for j, r_rec in enumerate(ref_records):
                r_str = " ".join([str(r_rec[r_mapping[k]]) for k in common_cols if pd.notna(r_rec[r_mapping[k]])])
                
                score = fuzz.ratio(s_str.lower(), r_str.lower())
                if score > best_score:
                    best_score = score
                    best_ref = j
                    
            if best_score >= threshold:
                matches.append({
                    "Source Index": i,
                    "Reference Index": best_ref,
                    "Confidence": round(best_score, 2),
                    "Status": "Match"
                })
                total_score += best_score
                match_count += 1
            else:
                matches.append({
                    "Source Index": i,
                    "Reference Index": None,
                    "Confidence": round(best_score, 2),
                    "Status": "Mismatch"
                })

        overall_score = round(total_score / match_count, 2) if match_count > 0 else 0.0
        return overall_score, pd.DataFrame(matches)
