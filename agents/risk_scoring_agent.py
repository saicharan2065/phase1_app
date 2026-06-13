class RiskScoringAgent:
    def __init__(self):
        pass

    def calculate_overall_risk(self, fraud_score=0, aml_score=0, graph_risk=0, entity_confidence=0, data_quality=100):
        # Weighting factors
        w_fraud = 0.4
        w_aml = 0.4
        w_graph = 0.2
        
        # Penalize for poor data quality
        dq_penalty = (100 - data_quality) * 0.1
        
        # Entity resolution confidence could amplify graph risk
        if entity_confidence > 0.8:
            graph_risk = min(graph_risk * 1.2, 100)
            
        overall_score = (fraud_score * w_fraud) + (aml_score * w_aml) + (graph_risk * w_graph) + dq_penalty
        overall_score = min(round(overall_score, 2), 100.0)
        
        category = "LOW"
        if overall_score >= 80: category = "CRITICAL"
        elif overall_score >= 60: category = "HIGH"
        elif overall_score >= 40: category = "MEDIUM"
        
        return overall_score, category
