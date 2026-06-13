from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
from graphs.graph_builder import EntityGraphEngine
from graphs.graph_visualizer import GraphVisualizer
from agents.risk_cluster_agent import RiskClusterDetector
import json

def build_and_visualize_graph(file):
    if file is None:
        return "Please upload a dataset.", "<h3>No graph</h3>", "{}", pd.DataFrame()
        
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file type.", "<h3>Error</h3>", "{}", pd.DataFrame()
            
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
            gr.Markdown("### Upload Dataset to Build Graph")
            ds_upload = gr.File(label="Dataset")
            dummy_btn = gr.Button('Generate Dummy Data', size='sm')
            dummy_btn.click(fn=lambda: generate_dummy_data('entity_graph'), outputs=ds_upload)
            build_btn = gr.Button("Generate Entity Graph", variant="primary")
            status_out = gr.Textbox(label="Status", interactive=False)
            stats_out = gr.Code(label="Graph Statistics", language="json")
            
        with gr.Column(scale=2):
            gr.Markdown("### Interactive Graph")
            graph_html = gr.HTML(label="Graph Visualization")
            
    gr.Markdown("### Detected Risk Clusters")
    clusters_table = gr.Dataframe(label="Fraud Rings / Suspicious Clusters")
            
    build_btn.click(
        fn=build_and_visualize_graph,
        inputs=ds_upload,
        outputs=[status_out, graph_html, stats_out, clusters_table]
    )
