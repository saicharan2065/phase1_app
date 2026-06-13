from utils.dummy_generator import generate_dummy_data
import gradio as gr
from agents.investigation_agent import InvestigationAgent
import json

def run_investigation(cust_id):
    if not cust_id:
        return "Please provide a Customer ID."
        
    agent = InvestigationAgent(graph=None) 
    result = agent.investigate(cust_id, fraud_score=60, aml_score=40)
    return json.dumps(result, indent=2)

def create_investigations_tab():
    gr.Markdown("### Investigation Agent")
    cust_id = gr.Textbox(label="Customer ID")
    inv_btn = gr.Button("Investigate", variant="primary")
    inv_out = gr.Code(label="Investigation Report", language="json")
    
    inv_btn.click(fn=run_investigation, inputs=cust_id, outputs=inv_out)
