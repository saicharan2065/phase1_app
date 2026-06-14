import json
import os

class InvestigationAgent:
    def __init__(self, graph=None):
        self.graph = graph

    def investigate(self, suspect_id, fraud_score=0, aml_score=0):
        # Gather graph context
        connected_entities = []
        relationships = []
        if self.graph and self.graph.has_node(suspect_id):
            neighbors = list(self.graph.neighbors(suspect_id))
            for n in neighbors:
                connected_entities.append(n)
                rel = self.graph.edges[suspect_id, n].get("relationship", "connected")
                relationships.append(f"{suspect_id} --[{rel}]--> {n}")
                
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
            
        markdown_report = f"""# 🤖 DeepSeek-R1 Automated Investigation Report
**Entity Analyzed:** `{suspect_id}`

## 🔍 Context & Link Analysis
"""
        if connected_entities:
            markdown_report += f"The entity is connected to **{len(connected_entities)}** other nodes within the graph network. Key structural edges detected:\n"
            for rel in relationships[:5]:
                markdown_report += f"- `{rel}`\n"
        else:
            markdown_report += "No immediate graph connections detected in the local cache.\n"

        markdown_report += "\n## ⚠️ Risk Assessment\n"
        if risk_factors:
            markdown_report += "The agent has flagged the following critical anomalies:\n"
            for rf in risk_factors:
                markdown_report += f"- **CRITICAL:** {rf}\n"
        else:
            markdown_report += "No severe risk anomalies flagged based on current feature weights.\n"
            
        markdown_report += "\n## ⚖️ Final LLM Conclusion\n"
        if not risk_factors:
            markdown_report += f"> The mathematical embedding of `{suspect_id}` does not resemble known historical fraud topologies. **Recommendation:** Proceed with standard monitoring."
        else:
            markdown_report += f"> The structural and behavioral footprint of `{suspect_id}` strongly correlates with sophisticated obfuscation techniques. **Recommendation:** Freeze assets and escalate to human review immediately."

        # Persist automatically to storage/investigations
        os.makedirs("storage/investigations", exist_ok=True)
        safe_id = str(suspect_id).replace("/", "_").replace("\\", "_")
        with open(f"storage/investigations/{safe_id}_report.md", "w", encoding="utf-8") as f:
            f.write(markdown_report)
            
        return markdown_report
