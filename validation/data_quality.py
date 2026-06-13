import pandas as pd
import numpy as np
import plotly.express as px

class DataQualityAnalyzer:
    def __init__(self):
        pass

    def analyze(self, df):
        if df is None or df.empty:
            return 0.0, pd.DataFrame(), None

        total_rows = len(df)
        total_cells = total_rows * len(df.columns)
        
        # Missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        # Duplicates
        duplicate_count = df.duplicated().sum()

        # Outliers (simple z-score for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        col_quality = []
        for col in df.columns:
            missing = missing_counts[col]
            missing_pct = (missing / total_rows) * 100
            
            outliers = 0
            if col in numeric_cols:
                # Handle standard deviation of 0
                std = df[col].std(ddof=0)
                if std > 0:
                    z_scores = np.abs((df[col] - df[col].mean()) / std)
                    outliers = (z_scores > 3).sum()
                    outlier_count += outliers
                
            score = max(0, 100 - missing_pct - (outliers / total_rows * 100))
            col_quality.append({
                "Column": col,
                "Missing (%)": round(missing_pct, 2),
                "Outliers": outliers,
                "Quality Score": round(score, 2)
            })

        col_quality_df = pd.DataFrame(col_quality)
        
        overall_score = max(0, 100 - (total_missing / total_cells * 100) - (duplicate_count / total_rows * 100))
        overall_score = round(overall_score, 2)

        # Generate visual
        fig = px.bar(col_quality_df, x="Column", y="Quality Score", title="Column Quality Scores", color="Quality Score", color_continuous_scale="RdYlGn")
        
        return overall_score, col_quality_df, fig
