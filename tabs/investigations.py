from utils.dummy_generator import generate_dummy_data
import gradio as gr
from agents.investigation_agent import InvestigationAgent
import json

import time

def run_investigation(suspect_id):
    if not suspect_id:
        yield "Please provide a Suspect ID."
        return
    try:
        agent = InvestigationAgent(graph=None) 
        result = agent.investigate(suspect_id, fraud_score=60, aml_score=40)
        yield result
    except Exception as e:
        yield f"🛑 CRITICAL PIPELINE ERROR: {str(e)}"

def create_investigations_tab(session_user=None):
    gr.Markdown("### Investigation Agent")
    suspect_id = gr.Textbox(label="Suspect ID")
    inv_btn = gr.Button("Initialize Autonomous Investigation", variant="primary")
    inv_out = gr.Markdown(label="Investigation Report")
    
    inv_btn.click(fn=run_investigation, inputs=suspect_id, outputs=inv_out)
