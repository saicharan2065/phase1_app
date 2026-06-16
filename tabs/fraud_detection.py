from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from models.fraud_detection import FraudDetectionEngine

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def run_fraud_detection(dataset_key):
    if not dataset_key or dataset_key not in GLOBAL_WORKSPACE_DATA:
        return "Please select a valid dataset from the workspace.", pd.DataFrame()
        
    try:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
        
        engine = FraudDetectionEngine()
        results_df = engine.detect_fraud(df)
        
        return "Fraud Detection Complete", results_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_fraud_detection_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Fraud Detection Engine")
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            run_btn = gr.Button("Run Detection", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Status")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### High-Risk Transactions / Entities")
    results_table = gr.Dataframe(label="Fraud Results")
            
    run_btn.click(fn=run_fraud_detection, inputs=ds_dropdown, outputs=[status_out, results_table])
