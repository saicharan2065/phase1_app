import gradio as gr
import pandas as pd
import json
from agents.schema_discovery_agent import SchemaDiscoveryAgent

def discover_schema_ui(file):
    if file is None:
        return "Please upload a dataset.", ""
        
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file type.", ""
            
        agent = SchemaDiscoveryAgent()
        mapping = agent.discover_schema(df.columns)
        
        return "Schema Discovery Complete", json.dumps(mapping, indent=2)
    except Exception as e:
        return f"Error: {str(e)}", ""

def create_schema_discovery_tab():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload Dataset (CSV/JSON/XLSX)")
            ds_upload = gr.File(label="Dataset")
            discover_btn = gr.Button("Discover Schema", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Discovered Schema Mapping")
            status_out = gr.Textbox(label="Status", interactive=False)
            mapping_out = gr.Code(label="Mapping (JSON)", language="json")
            
    discover_btn.click(fn=discover_schema_ui, inputs=ds_upload, outputs=[status_out, mapping_out])
