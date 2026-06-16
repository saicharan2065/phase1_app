from utils.dummy_generator import generate_dummy_data
import gradio as gr
import pandas as pd
import json
from agents.schema_discovery_agent import SchemaDiscoveryAgent

from data.dataset_manager import get_user_workspace

def discover_schema_ui(dataset_key, file, username):
    if file is not None:
        try:
            if file.name.endswith('.csv'): df = pd.read_csv(file.name)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file.name)
            elif file.name.endswith('.json'): df = pd.read_json(file.name)
            else: return "Unsupported file type.", ""
        except Exception as e:
            return f"File error: {e}", ""
    elif dataset_key and dataset_key in get_user_workspace(username):
        df = get_user_workspace(username)[dataset_key]
    else:
        return "Please select a valid dataset from the workspace, or upload a file.", ""
        
    try:
        agent = SchemaDiscoveryAgent()
        mapping = agent.discover_schema(df.columns)
        
        return "Schema Discovery Complete", json.dumps(mapping, indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""

def create_schema_discovery_tab(session_user):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Select Workspace Dataset")
            with gr.Row():
                ds_dropdown = gr.Dropdown(choices=[], label="Dataset", scale=4)
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                
            gr.Markdown("### Or Upload Direct File")
            ds_upload = gr.File(label="Dataset Fallback")
            
            discover_btn = gr.Button("Discover Schema", variant="primary")
            
            refresh_btn.click(fn=lambda u: gr.update(choices=list(get_user_workspace(u).keys())), inputs=session_user, outputs=ds_dropdown)
            
        with gr.Column():
            gr.Markdown("### Discovered Schema Mapping")
            status_out = gr.Textbox(label="Status", interactive=False)
            mapping_out = gr.Code(label="Mapping (JSON)", language="json")
            
    discover_btn.click(fn=discover_schema_ui, inputs=[ds_dropdown, ds_upload, session_user], outputs=[status_out, mapping_out])
