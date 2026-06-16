import pandas as pd

class AMLDetectionEngine:
    def __init__(self):
        pass

    def detect_aml(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
            
        results = []
        id_col = next((c for c in df.columns if 'id' in c.lower() or 'account' in c.lower()), df.columns[0])
        amount_col = next((c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower()), None)
        
        # Group by ID to detect patterns
        if amount_col:
            grouped = df.groupby(id_col)
            for entity_id, group in grouped:
                reasons = []
                score = 0
                
                # Structuring / Smurfing: many transactions just below 10,000 threshold
                structuring_txns = group[(group[amount_col] >= 9000) & (group[amount_col] < 10000)]
                if len(structuring_txns) >= 2:
                    reasons.append("Potential Structuring (Multiple Txns just below 10k)")
                    score += 40
                    
                # Round-dollar patterns
                round_txns = group[group[amount_col] % 1000 == 0]
                if len(round_txns) >= 3:
                    reasons.append("Round-dollar pattern detected")
                    score += 20
                    
                # High velocity / Rapid movement
                if len(group) >= 10:
                    reasons.append("Rapid movement of funds (High Velocity)")
                    score += 30
                    
                score = min(score, 100)
                
                if score > 0:
                    results.append({
                        "Entity ID": entity_id,
                        "AML Score": score,
                        "AML Alert": "CRITICAL" if score > 75 else ("HIGH" if score > 50 else "REVIEW"),
                        "Reasons": "; ".join(reasons)
                    })
                    
        df_results = pd.DataFrame(results)
        if df_results.empty:
            return pd.DataFrame([{"Message": "Clean data, no AML risk patterns detected."}])
        return df_results.sort_values("AML Score", ascending=False)
