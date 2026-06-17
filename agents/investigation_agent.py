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
        
        summary = f"Investigation for {suspect_id} completed. "
        if not risk_factors:
            summary += "No significant anomalies detected."
        else:
            summary += "Multiple high-risk signals present requiring manual review."
            
        try:
            import torch
            from agents.vram_manager import vram_manager
            from tabs.model_management import get_active_model_state
            
            active_model = get_active_model_state()
            model_id = active_model if active_model and active_model != "None Selected" else "Qwen/Qwen1.5-0.5B"
            
            model, tokenizer = vram_manager.get_or_load_model(model_id, use_4bit=True)
            
            prompt = f"Write a detailed forensic investigation report for suspect {suspect_id}. They have the following risk factors: {', '.join(risk_factors)}. Start the report with a markdown header."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
                
            markdown_report = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            raise RuntimeError(f"MI300X REAL LLM INFERENCE FAILED: {str(e)}")

        # Persist automatically to storage/investigations
        os.makedirs("storage/investigations", exist_ok=True)
        safe_id = str(suspect_id).replace("/", "_").replace("\\", "_")
        with open(f"storage/investigations/{safe_id}_report.md", "w", encoding="utf-8") as f:
            f.write(markdown_report)
            
        return markdown_report
