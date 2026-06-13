import gradio as gr
import pandas as pd
from agents.fraud_ring_detector import FraudRingDetector
from graphs.graph_builder import EntityGraphEngine

def detect_rings(file):
    if file is None:
        return "Upload a dataset.", pd.DataFrame()
        
    try:
        if file.name.endswith('.csv'): df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
        elif file.name.endswith('.json'): df = pd.read_json(file.name)
        else: return "Unsupported file.", pd.DataFrame()
        
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
            ds_upload = gr.File(label="Dataset")
            run_btn = gr.Button("Detect Rings", variant="primary")
            status_out = gr.Textbox(label="Status", interactive=False)
            
    gr.Markdown("### Detected Fraud Rings")
    results_table = gr.Dataframe(label="Risk Clusters")
            
    run_btn.click(fn=detect_rings, inputs=ds_upload, outputs=[status_out, results_table])
