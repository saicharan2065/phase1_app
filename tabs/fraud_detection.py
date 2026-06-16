from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from models.fraud_detection import FraudDetectionEngine

from data.dataset_manager import get_user_workspace

def run_fraud_detection(dataset_key, file, username):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", pd.DataFrame()
        except Exception as e:
            return f"File error: {e}", pd.DataFrame()
    elif dataset_key and dataset_key in get_user_workspace(username):
        df = get_user_workspace(username)[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", pd.DataFrame()
        
    try:
        
        engine = FraudDetectionEngine()
        results_df = engine.detect_fraud(df)
        
        return "Fraud Detection Complete", results_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_fraud_detection_tab(session_user):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Fraud Detection Engine")
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=[], label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            run_btn = gr.Button("Run Detection", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Status")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### High-Risk Transactions / Entities")
    results_table = gr.Dataframe(label="Fraud Results")
            
    run_btn.click(fn=run_fraud_detection, inputs=[ds_dropdown, ds_upload, session_user], outputs=[status_out, results_table])
