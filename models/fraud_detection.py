import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import os

class FraudDetectionEngine:
    def __init__(self, contamination=0.01):
        self.contamination = contamination
        self.if_model = IsolationForest(contamination=self.contamination, random_state=42)
        
    def detect_fraud(self, df):
        if df is None or df.empty:
            return pd.DataFrame()
            
        results = []
        
        # Determine relevant columns
        id_col = next((c for c in df.columns if 'id' in c.lower()), df.columns[0])
        amount_col = next((c for c in df.columns if 'amount' in c.lower() or 'amt' in c.lower() or 'value' in c.lower()), None)
        merchant_col = next((c for c in df.columns if 'merchant' in c.lower()), None)
        country_col = next((c for c in df.columns if 'country' in c.lower()), None)
        
        # Simple velocity checks and large amounts
        mean_amt = df[amount_col].mean() if amount_col else 0
        std_amt = df[amount_col].std(ddof=0) if amount_col else 0

        # Anomaly Detection using Isolation Forest if amount is present
        if amount_col:
            # Dropna for modeling
            X = df[[amount_col]].fillna(0)
            if len(X) > 10:
                df['if_score'] = self.if_model.fit_predict(X)
                df['lof_score'] = LocalOutlierFactor(contamination=self.contamination).fit_predict(X)

        for idx, row in df.iterrows():
            reasons = []
            risk_score = 0
            
            # Statistical Outlier
            if amount_col and pd.notna(row[amount_col]):
                amt = row[amount_col]
                if amt > mean_amt + 3 * std_amt and std_amt > 0:
                    reasons.append("High Amount Statistical Outlier")
                    risk_score += 40
            
            # Isolation Forest anomaly
            if 'if_score' in df.columns and row.get('if_score') == -1:
                reasons.append("Anomaly detected via Isolation Forest")
                risk_score += 30
                
            if 'lof_score' in df.columns and row.get('lof_score') == -1:
                reasons.append("Anomaly detected via Local Outlier Factor")
                risk_score += 20
                
            # Country risk (mock logic)
            if country_col and pd.notna(row[country_col]) and str(row[country_col]).upper() in ["SY", "KP", "IR", "CU"]:
                reasons.append("High Risk Country")
                risk_score += 50
                
            # Merchant abuse (mock logic)
            if merchant_col and pd.notna(row[merchant_col]) and "CASINO" in str(row[merchant_col]).upper():
                reasons.append("High Risk Merchant Category")
                risk_score += 20
                
            risk_score = min(risk_score, 100)
            
            category = "LOW"
            if risk_score > 75: category = "CRITICAL"
            elif risk_score > 50: category = "HIGH"
            elif risk_score > 25: category = "MEDIUM"
                
            if risk_score > 0:
                results.append({
                    "Entity ID": row[id_col],
                    "Risk Score": risk_score,
                    "Risk Category": category,
                    "Fraud Reason": "; ".join(reasons)
                })
                
        if not results:
            return pd.DataFrame([{"Message": "Clean data, no fraud detected."}])
            
        return pd.DataFrame(results).sort_values("Risk Score", ascending=False)
