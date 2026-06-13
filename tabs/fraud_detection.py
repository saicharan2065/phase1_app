from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from models.fraud_detection import FraudDetectionEngine

def run_fraud_detection(file):
    if file is None:
        return "Upload a dataset.", pd.DataFrame()
        
    try:
        if file.name.endswith('.csv'): df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
        elif file.name.endswith('.json'): df = pd.read_json(file.name)
        else: return "Unsupported file.", pd.DataFrame()
        
        engine = FraudDetectionEngine()
        results_df = engine.detect_fraud(df)
        
        return "Fraud Detection Complete", results_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_fraud_detection_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Fraud Detection Engine")
            ds_upload = gr.File(label="Dataset")
            dummy_btn = gr.Button('Generate Dummy Data', size='sm')
            dummy_count = gr.Dropdown(choices=["15", "100", "500", "1000", "5000", "10000"], value="15", label="Records to Generate")
            dummy_btn.click(fn=lambda n: generate_dummy_data("fraud", int(n)), inputs=dummy_count, outputs=ds_upload)
            run_btn = gr.Button("Run Detection", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Status")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### High-Risk Transactions / Entities")
    results_table = gr.Dataframe(label="Fraud Results")
            
    run_btn.click(fn=run_fraud_detection, inputs=ds_upload, outputs=[status_out, results_table])
