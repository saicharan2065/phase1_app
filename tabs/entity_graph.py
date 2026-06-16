from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from graphs.graph_builder import EntityGraphEngine
from graphs.graph_visualizer import GraphVisualizer
from agents.risk_cluster_agent import RiskClusterDetector
import json

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def generate_graph(dataset_key, file):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", "<h3>Error</h3>", "{}", pd.DataFrame()
        except Exception as e:
            return f"File error: {e}", f"<h3>Error: {e}</h3>", "{}", pd.DataFrame()
    elif dataset_key and dataset_key in GLOBAL_WORKSPACE_DATA:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", "<h3>No graph</h3>", "{}", pd.DataFrame()
        
    try:
        # Build Graph
        engine = EntityGraphEngine()
        engine.build_from_dataframe(df)
        
        # Visualize
        viz = GraphVisualizer(engine.graph)
        html_content = viz.generate_html()
        
        # Statistics
        stats = engine.get_statistics()
        
        # Risk Clusters
        risk_detector = RiskClusterDetector(engine.graph)
        clusters = risk_detector.detect_clusters()
        clusters_df = pd.DataFrame(clusters)
        
        return "Graph Generation Complete", html_content, json.dumps(stats, indent=2), clusters_df
    except Exception as e:
        return f"Error: {str(e)}", f"<h3>Error: {str(e)}</h3>", "{}", pd.DataFrame()

def create_entity_graph_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            gen_btn = gr.Button("Generate Graph", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            status_out = gr.Textbox(label="Status", interactive=False)
            stats_out = gr.Code(label="Graph Statistics", language="json")
            
        with gr.Column(scale=2):
            gr.Markdown("### Interactive Graph")
            graph_html = gr.HTML(label="Graph Visualization")
            
    gr.Markdown("### Detected Risk Clusters")
    clusters_table = gr.Dataframe(label="Fraud Rings / Suspicious Clusters")
            
    gen_btn.click(
        fn=generate_graph,
        inputs=[ds_dropdown, ds_upload],
        outputs=[status_out, graph_html, stats_out, clusters_table]
    )
