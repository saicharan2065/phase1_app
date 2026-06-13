from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from models.aml_detection import AMLDetectionEngine

def run_aml_detection(file):
    if file is None:
        return "Upload a dataset.", pd.DataFrame()
        
    try:
        if file.name.endswith('.csv'): df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
        elif file.name.endswith('.json'): df = pd.read_json(file.name)
        else: return "Unsupported file.", pd.DataFrame()
        
        engine = AMLDetectionEngine()
        results_df = engine.detect_aml(df)
        
        return "AML Detection Complete", results_df
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_aml_detection_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### AML Detection Engine")
            ds_upload = gr.File(label="Dataset")
            dummy_btn = gr.Button('Generate Dummy Data', size='sm')
            dummy_btn.click(fn=lambda: generate_dummy_data('aml'), outputs=ds_upload)
            run_btn = gr.Button("Run Detection", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Status")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### AML Alerts")
    results_table = gr.Dataframe(label="AML Results")
            
    run_btn.click(fn=run_aml_detection, inputs=ds_upload, outputs=[status_out, results_table])
