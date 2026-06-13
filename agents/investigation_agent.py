import json
import os

class InvestigationAgent:
    def __init__(self, graph=None):
        self.graph = graph

    def investigate(self, customer_id, fraud_score=0, aml_score=0):
        # Gather graph context
        connected_entities = []
        relationships = []
        if self.graph and self.graph.has_node(customer_id):
            neighbors = list(self.graph.neighbors(customer_id))
            for n in neighbors:
                connected_entities.append(n)
                rel = self.graph.edges[customer_id, n].get("relationship", "connected")
                relationships.append(f"{customer_id} --[{rel}]--> {n}")
                
        # Aggregate Risk Factors
        risk_factors = []
        if fraud_score > 50: risk_factors.append(f"High Fraud Score ({fraud_score})")
        if aml_score > 50: risk_factors.append(f"High AML Score ({aml_score})")
        if len(connected_entities) > 5: risk_factors.append(f"High number of shared connections ({len(connected_entities)})")
        
        summary = f"Investigation for {customer_id} completed. "
        if not risk_factors:
            summary += "No significant anomalies detected."
        else:
            summary += "Multiple high-risk signals present requiring manual review."
            
        result = {
            "Customer ID": customer_id,
            "Connected Entities": connected_entities,
            "Graph Relationships": relationships,
            "Risk Factors": risk_factors,
            "Investigation Summary": summary
        }
        
        # Persist automatically to storage/investigations
        os.makedirs("storage/investigations", exist_ok=True)
        # Sanitizing customer ID for file path
        safe_id = str(customer_id).replace("/", "_").replace("\\", "_")
        with open(f"storage/investigations/{safe_id}_report.json", "w") as f:
            json.dump(result, f, indent=4)
            
        return result
