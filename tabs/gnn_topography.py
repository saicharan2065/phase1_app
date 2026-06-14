import gradio as gr
import time
import pandas as pd
import threading
from agents.gnn_engine import GNNEngine

gnn_engine = GNNEngine()

def run_gnn_analytics():
    if gnn_engine.is_running:
        yield "Already computing...", pd.DataFrame()
        return
        
    yield "Initializing 240GB RAM Allocation...", pd.DataFrame()
    
    t = threading.Thread(target=gnn_engine.run_deep_graph_analytics)
    t.start()
    
    while True:
        time.sleep(0.5)
        bar_len = int((gnn_engine.processed_nodes / gnn_engine.total_nodes) * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        # Format large numbers for readability
        processed = f"{gnn_engine.processed_nodes:,}"
        total = f"{gnn_engine.total_nodes:,}"
        
        progress = f"{gnn_engine.status_message}\n[{bar}] {processed} / {total} Nodes Computed"
        
        current_df = pd.DataFrame(gnn_engine.findings)
        yield progress, current_df
        
        if not gnn_engine.is_running and gnn_engine.processed_nodes >= gnn_engine.total_nodes:
            break
            
    t.join()
    final_df = pd.DataFrame(gnn_engine.findings)
    yield f"COMPLETE: Analyzed {total} transaction nodes across Global Financial System.", final_df

def create_gnn_topography_tab():
    gr.Markdown("### 🕸️ Deep Graph Convolutional Networks (GNN)")
    gr.Markdown("Utilize your **240 GB of System RAM** to construct massive Adjacency Matrices of the global financial network. The Matrix is then pushed into the **MI300X 192GB VRAM** to compute Graph Embeddings that mathematically identify hidden fraud clusters across **500 Million** nodes.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Global Financial Graph")
            gr.Markdown("**Input Target:** `5TB NVMe Global Ledger DB`")
            start_btn = gr.Button("🚀 Launch 500M Node GNN", variant="primary")
            status_out = gr.Textbox(label="GNN Telemetry", lines=4, interactive=False)
            
        with gr.Column(scale=2):
            results_table = gr.Dataframe(label="Anomalous Topology Clusters Discovered", max_height=300)
            
    start_btn.click(fn=run_gnn_analytics, outputs=[status_out, results_table])
