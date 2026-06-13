import pandas as pd
from rapidfuzz import fuzz

class EntityResolutionAgent:
    def __init__(self, threshold=85.0):
        self.threshold = threshold

    def resolve_entities(self, df):
        results = []
        if df is None or df.empty or len(df) < 2:
            return results
            
        id_col = next((c for c in df.columns if 'id' in c.lower()), df.columns[0])
        name_col = next((c for c in df.columns if 'name' in c.lower()), None)
        phone_col = next((c for c in df.columns if 'phone' in c.lower() or 'mobile' in c.lower()), None)
        email_col = next((c for c in df.columns if 'email' in c.lower()), None)

        records = df.to_dict('records')
        
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                rec1 = records[i]
                rec2 = records[j]
                
                # Check for exact matches
                if phone_col and pd.notna(rec1.get(phone_col)) and rec1.get(phone_col) == rec2.get(phone_col):
                    results.append({
                        "entity_1": str(rec1.get(id_col, i)),
                        "entity_2": str(rec2.get(id_col, j)),
                        "confidence": 1.0,
                        "reason": "phone_match"
                    })
                    continue
                    
                if email_col and pd.notna(rec1.get(email_col)) and rec1.get(email_col) == rec2.get(email_col):
                    results.append({
                        "entity_1": str(rec1.get(id_col, i)),
                        "entity_2": str(rec2.get(id_col, j)),
                        "confidence": 1.0,
                        "reason": "email_match"
                    })
                    continue
                
                # Check fuzzy name matching
                if name_col and pd.notna(rec1.get(name_col)) and pd.notna(rec2.get(name_col)):
                    score = fuzz.ratio(str(rec1.get(name_col)).lower(), str(rec2.get(name_col)).lower())
                    if score >= self.threshold:
                        results.append({
                            "entity_1": str(rec1.get(id_col, i)),
                            "entity_2": str(rec2.get(id_col, j)),
                            "confidence": round(score / 100.0, 2),
                            "reason": "fuzzy_name_match"
                        })
                        
        return results
