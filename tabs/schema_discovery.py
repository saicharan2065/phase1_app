from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
import json
from agents.schema_discovery_agent import SchemaDiscoveryAgent

from data.dataset_manager import GLOBAL_WORKSPACE_DATA

def discover_schema_ui(dataset_key):
    if not dataset_key or dataset_key not in GLOBAL_WORKSPACE_DATA:
        return "Please select a valid dataset from the workspace.", ""
        
    try:
        df = GLOBAL_WORKSPACE_DATA[dataset_key]
        agent = SchemaDiscoveryAgent()
        mapping = agent.discover_schema(df.columns)
        
        return "Schema Discovery Complete", json.dumps(mapping, indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""

def create_schema_discovery_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=list(GLOBAL_WORKSPACE_DATA.keys()), label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
            discover_btn = gr.Button("Discover Schema", variant="primary")
            
            refresh_btn.click(fn=lambda: gr.update(choices=list(GLOBAL_WORKSPACE_DATA.keys())), outputs=ds_dropdown)
            
        with gr.Column():
            gr.Markdown("### Discovered Schema Mapping")
            status_out = gr.Textbox(label="Status", interactive=False)
            mapping_out = gr.Code(label="Mapping (JSON)", language="json")
            
    discover_btn.click(fn=discover_schema_ui, inputs=ds_dropdown, outputs=[status_out, mapping_out])
