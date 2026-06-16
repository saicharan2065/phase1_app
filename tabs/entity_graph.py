from utils.dummy_generator import generate_dummy_data
import gradio as gr
from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from graphs.graph_builder import EntityGraphEngine
from graphs.graph_visualizer import GraphVisualizer
from agents.risk_cluster_agent import RiskClusterDetector
import json

from data.dataset_manager import get_user_workspace

def generate_graph(dataset_key, file, username):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", "<h3>Error</h3>", "{}", pd.DataFrame()
        except Exception as e:
            return f"File error: {e}", f"<h3>Error: {e}</h3>", "{}", pd.DataFrame()
    elif dataset_key and dataset_key in get_user_workspace(username):
        df = get_user_workspace(username)[dataset_key]
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

def create_entity_graph_tab(session_user):
    gr.Markdown("### 🕸️ Entity Graph Network Builder")
    gr.Markdown("An **Entity Graph** is a network visualization where points (nodes) represent people, companies, or accounts, and the lines (edges) connecting them represent transactions or relationships. This tool scans your dataset to link related entities, making it incredibly easy to visualize hidden 'Fraud Rings' or cyclic money laundering networks that are impossible to spot in a spreadsheet.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=[], label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            gen_btn = gr.Button("Generate Graph", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            status_out = gr.Textbox(label="Status", interactive=False)
            stats_out = gr.Code(label="Graph Statistics", language="json")
            
        with gr.Column(scale=2):
            gr.Markdown("### Interactive Graph")
            graph_html = gr.HTML(label="Graph Visualization")
            
    gr.Markdown("### Detected Risk Clusters")
    clusters_table = gr.Dataframe(label="Fraud Rings / Suspicious Clusters")
            
    gen_btn.click(
        fn=generate_graph,
        inputs=[ds_dropdown, ds_upload, session_user],
        outputs=[status_out, graph_html, stats_out, clusters_table]
    )
