from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from agents.fraud_ring_detector import FraudRingDetector
from graphs.graph_builder import EntityGraphEngine

from data.dataset_manager import get_user_workspace

def detect_rings(dataset_key, file, username):
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
        
        engine = EntityGraphEngine()
        engine.build_from_dataframe(df)
        
        detector = FraudRingDetector(engine.graph)
        clusters = detector.detect_rings()
        
        return "Risk Cluster Detection Complete", pd.DataFrame(clusters)
    except Exception as e:
        return f"Error: {str(e)}", pd.DataFrame()

def create_risk_clusters_tab(session_user):
    gr.Markdown("### 🕸️ Risk Cluster & Fraud Ring Detector")
    gr.Markdown("This engine scans your underlying Entity Graph to detect highly connected **Fraud Rings**. It looks for groups of accounts transacting in tight circles (cyclic flows), high-risk multi-hop relationships, and known bad actors forming suspicious clusters.")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=[], label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            run_btn = gr.Button("Detect Rings", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### Detected Fraud Rings")
    results_table = gr.Dataframe(label="Risk Clusters")
            
    run_btn.click(fn=detect_rings, inputs=[ds_dropdown, ds_upload, session_user], outputs=[status_out, results_table])
