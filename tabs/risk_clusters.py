from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from agents.fraud_ring_detector import FraudRingDetector
from graphs.graph_builder import EntityGraphEngine

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def detect_rings(dataset_key, file):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", pd.DataFrame()
        except Exception as e:
            return f"File error: {e}", pd.DataFrame()
    elif dataset_key and dataset_key in GLOBAL_WORKSPACE_DATA:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", pd.DataFrame()
        
    try:
        
        engine = EntityGraphEngine()
        engine.build_from_dataframe(df)
        
        detector = FraudRingDetector(engine.graph)
        clusters = detector.detect_rings()
        
        return "Risk Cluster Detection Complete", pd.DataFrame(clusters)
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_risk_clusters_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Fraud Ring Detector")
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            run_btn = gr.Button("Detect Rings", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### Detected Fraud Rings")
    results_table = gr.Dataframe(label="Risk Clusters")
            
    run_btn.click(fn=detect_rings, inputs=[ds_dropdown, ds_upload], outputs=[status_out, results_table])
