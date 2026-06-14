from utils.dummy_generator import generate_dummy_data
import gradio as gr
from agents.investigation_agent import InvestigationAgent
import json

import time

def run_investigation(cust_id):
    if not cust_id:
        yield "Please provide a Customer ID."
        return
        
    agent = InvestigationAgent(graph=None) 
    result = agent.investigate(cust_id, fraud_score=60, aml_score=40)
    
    output = ""
    # Simulate LLM typing effect
    for chunk in result.split(" "):
        output += chunk + " "
        yield output
        time.sleep(0.05)

def create_investigations_tab():
    gr.Markdown("### Investigation Agent")
    cust_id = gr.Textbox(label="Customer ID")
    inv_btn = gr.Button("Initialize Autonomous Investigation", variant="primary")
    inv_out = gr.Markdown(label="Investigation Report")
    
    inv_btn.click(fn=run_investigation, inputs=cust_id, outputs=inv_out)
